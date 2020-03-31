import argparse
import copy
import os
from glob import glob
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import traceback
from itertools import accumulate as accumulate
from model import *
from loss import *


def save_losses(losses, destination):
    plt.clf()
    if losses.shape[0] < 8:
        colors = ['red', 'blue', 'yellow', 'black', 'green', 'magenta', 'cyan']
        #markers = ['o', 's', '*', '+', 'D', '|', '_']
        markers = [''] * 7
    else:
        colors = ['black'] * losses.shape[0]
        markers = [''] * losses.shape[0]
    losses = losses.view(losses.size(0), losses.size(1), -1)
    torch.save(losses.detach(), destination+'.pt')
    losses = losses.numpy()

    for i in range(losses.shape[0]):
        plt.plot(range(losses.shape[1]), losses[i, :, 0], color=colors[i], marker=markers[i], linestyle='solid')
        if losses.shape[2] > 1:
            plt.plot(range(losses.shape[1]), losses[i, :, 1], color=colors[i], marker=markers[i], linestyle='dashed')
        if losses.shape[2] > 2:
            plt.plot(range(losses.shape[1]), losses[i, :, 2], color=colors[i], marker=markers[i], linestyle='dotted')

    plt.savefig(destination+'.png')
    plt.clf()


def build_models(args, inputs):
    class_ = FactoredModel if args.factored_model else ConvolutionalModel
    data_size = inputs.size(0)
    partitions = partition_data(data_size, args.partitions)

    if args.partitions > 1:
        weightor = [ClassificationModel(inputs.mean(dim=0), inputs.std(dim=0), args.partitions).cuda()]
    else:
        weightor = []
    predictors = []
    for partition in partitions:
        train_inputs = None
        if args.partition_training:
            train_inputs = inputs[partition[1]]
        else:
            train_inputs = torch.cat([inputs[partition[0]], inputs[partition[2]]], dim=0)
        predictors += [class_(train_inputs.mean(dim=0), train_inputs.std(dim=0)).cuda()]
    return predictors + weightor


def partition_data(dsize, n_partitions):
    partition_size = dsize // n_partitions
    partition_sizes = [partition_size] * (n_partitions - 1) + [dsize - (partition_size * (n_partitions - 1))]
    partition_edges = [0] + list(accumulate(partition_sizes))
    partition_slices = [(slice(0, partition_edges[i]), slice(partition_edges[i], partition_edges[i+1]), slice(partition_edges[i+1], None)) for i in range(n_partitions)]
    return partition_slices


def shuffle_data(*tensors):
    i_shuffle = torch.randperm(tensors[0].size(0))
    return [tensors[i][i_shuffle] for i in range(len(tensors))]

'''
Performs a single gradient update and returns losses
'''
def step(model, inputs, outputs, loss_f, opt):
    predictions = model(inputs)#torch.sigmoid(model(inputs))#
    loss = loss_f(predictions.squeeze(), outputs.squeeze())
    torch.mean(loss).backward()
    opt.step()
    opt.zero_grad()
    return loss


def build_resample_probs(outputs):
    normal_ = torch.randn(outputs.size(0)) * torch.std(outputs) + torch.mean(outputs)
    outputs_histogram = plt.hist(outputs, 1000, (-5, 5))
    normal_histogram = plt.hist(normal_, 1000, (-5, 5))
    new_histogram = normal_histogram[0] / (outputs_histogram[0] + 1e-6)
    new_histogram = new_histogram / (np.sum(new_histogram * outputs_histogram[0]))

    truncated_outputs = outputs[outputs > -5]
    truncated_outputs = truncated_outputs[truncated_outputs < 5]
    probabilities = torch.zeros(truncated_outputs.shape)

    for idx, edges in enumerate(np.stack([normal_histogram[1][:-1], normal_histogram[1][1:]], axis=1)):
        probabilities[(truncated_outputs >= edges[0]) * (truncated_outputs <= edges[1])] = new_histogram[idx]

    return probabilities


def train_model(model, inputs, outputs, partition, loss_f, eval_f, opt, epochs, args, model_name='model'):
    partition_inputs = inputs[partition[1]]
    partition_outputs = outputs[partition[1]]
    reverse_inputs = torch.cat([inputs[partition[0]], inputs[partition[2]]])
    reverse_outputs = torch.cat([outputs[partition[0]], outputs[partition[2]]])

    if args.partition_training:
        train_inputs = partition_inputs
        train_outputs = partition_outputs
        eval_inputs = reverse_inputs
        eval_outputs = reverse_outputs
    else:
        train_inputs = reverse_inputs
        train_outputs = reverse_outputs
        eval_inputs = partition_inputs
        eval_outputs = partition_outputs

    print(train_inputs.size())
    print(eval_inputs.size())

    # Resample train set to a normal distribution over targets
    if isinstance(outputs, torch.FloatTensor):
        resample_probs = build_resample_probs(train_outputs)
        resample_indices = torch.multinomial(resample_probs, train_outputs.size(0), replacement=True)
    trainval_inputs = copy.deepcopy(train_inputs)
    trainval_outputs = copy.deepcopy(train_outputs)
        #train_inputs = train_inputs[resample_indices]
        #train_outputs = train_outputs[resample_indices]

    # Pre-calculate the mean of the eval data and from it the error for R^2
    try:
        data_mean = torch.mean(eval_outputs).detach()
        data_error = eval_f(eval_outputs.detach(), torch.ones(eval_outputs.size(0))*data_mean).detach() / eval_outputs.size(0)
    except:
        #data_mean = torch.mode(eval_outputs, dim=0).values.view(-1).detach()
        data_error = torch.FloatTensor([1]).to(eval_inputs)

    n_batches = (train_inputs.size(0) // args.batch_size)+1
    train_losses = torch.zeros(n_batches)
    losses = torch.zeros(args.epochs, 3)

    # Initialize logit tensor for loss-based importance sampling
    if args.loss_sampling:
        logits = torch.zeros(train_inputs.size(0))

    # Run train/eval loop over specified number of epochs
    for i_epoch in range(args.epochs):
        # Increase batch size according to specified schedule
        args.batch_size = int(args.batch_size + args.batch_size_annealing)
        if i_epoch < (args.epochs - epochs):
            continue
        n_batches = (train_inputs.size(0) // args.batch_size)

        # Set model to training mode
        model.train()

        # Shuffle training data so it is seen in a different order each epoch
        if not args.loss_sampling:
            train_inputs, train_outputs = shuffle_data(train_inputs, train_outputs)

        # Batch training data
        for i_batch in range(n_batches):
            # Build batches
            if args.loss_sampling and i_epoch > 0:
                batch_indices = torch.multinomial(logits, args.batch_size, replacement=True)
            else:
                batch_indices = slice(i_batch*args.batch_size, (i_batch+1)*args.batch_size)

            batch_inputs = train_inputs[batch_indices].cuda()
            batch_outputs = train_outputs[batch_indices].cuda()

            # If the last batch is size 0, just skip it
            if batch_inputs.size(0) == 0:
                continue

            # Perform gradient update on batch
            batch_losses = step(model, batch_inputs, batch_outputs, loss_f, opt)
            train_losses[i_batch] = torch.mean(batch_losses).detach().cpu()

            # If using loss sampling, update logits
            if args.loss_sampling:
                with torch.no_grad():
                    logits[batch_indices] = batch_losses.cpu()

        # Set model to evaluation mode (turn off dropout and stuff)
        model.eval()

        n_batches = (eval_inputs.size(0) // args.batch_size)+1
        sum_loss = 0

        # Batch the eval data
        eval_inputs, eval_outputs = shuffle_data(eval_inputs, eval_outputs)
        for i_batch in range(25):
            batch_inputs = eval_inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
            batch_outputs = eval_outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()

            # Same reasoning as training: sometimes encounter 0-size batches
            if batch_inputs.size(0) == 0:
                continue

            # Don't need to track operations/gradients for evaluation
            with torch.no_grad():
                # Build a sum of evaluation losses to average over later
                predictions = model(batch_inputs, eval=True).squeeze()#torch.sigmoid(model(batch_inputs, eval=True).squeeze())
                sum_loss += eval_f(predictions.squeeze(), batch_outputs.squeeze()).item()

        n_batches = (trainval_inputs.size(0) // args.batch_size)+1
        sum_loss2 = 0
        # Batch the eval data
        trainval_inputs, trainval_outputs = shuffle_data(trainval_inputs, trainval_outputs)
        for i_batch in range(25):
            batch_inputs = trainval_inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
            batch_outputs = trainval_outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()

            # Same reasoning as training: sometimes encounter 0-size batches
            if batch_inputs.size(0) == 0:
                continue

            # Don't need to track operations/gradients for evaluation
            with torch.no_grad():
                # Build a sum of evaluation losses to average over later
                predictions = model(batch_inputs, eval=True).squeeze()#torch.sigmoid(model(batch_inputs, eval=True).squeeze())
                sum_loss2 += eval_f(predictions.squeeze(), batch_outputs.squeeze()).item()

        # Calculate and print mean train and eval loss over the epoch
        mean_loss = sum_loss / (args.batch_size * 25)#eval_inputs.size(0)#
        mean_loss2 = sum_loss2 / (args.batch_size * 25)#trainval_inputs.size(0)#
        losses[i_epoch, 0] = torch.mean(train_losses)
        losses[i_epoch, 1] = mean_loss
        losses[i_epoch, 2] = mean_loss2
        print('Epoch %d Mean Train / TrainVal / Eval Loss and R^2 Value: %.3f / %.3f / %.3f / %.3f ' % (i_epoch+1, losses[i_epoch, 0], losses[i_epoch, 2], losses[i_epoch, 1], 1 - (mean_loss / data_error).item()), end='\r')

        if (i_epoch+1) % args.save_rate == 0:
            torch.save(model.cpu(), args.model_path + '/' + model_name + '_%d.ptm' % (i_epoch+1))
            model.cuda()

    print('') # to keep only the final epoch losses from each fold
    return model, losses.cpu()


def train(models, inputs, outputs, test_inputs, test_outputs, epochs, args):
    # Separate loss functions for training and evaluation, allows for
    # potentially more flexible evaluation metrics
    loss_f = nn.MSELoss(reduction='none')
    eval_f = nn.MSELoss(reduction='sum')
    if args.factored_model:
        loss_f = FactoredLoss(reduction=False)

    if args.partitions > 1:
        print('Training an ensemble model via %d-fold cross-validation.' % args.partitions)
        return cross_validation(models, inputs, outputs, test_inputs, test_outputs, loss_f, eval_f, epochs, args)
    else:
        print('Training a single model on train and test data.')
        args.partition_training = False
        partition = [slice(train_inputs.size(0)), slice(train_inputs.size(0), None), slice(0)]
        inputs = torch.cat([train_inputs, test_inputs], dim=0)
        outputs = torch.cat([train_outputs, test_outputs], dim=0)
        opt = optim.Adam(models[0].parameters(), lr=args.learning_rate)
        model, losses = train_model(models[0], inputs, outputs, partition, loss_f, eval_f, opt, args)
        save_losses(losses.cpu(), args.model_path+'/losses')
        return model, losses.cpu()


def cross_validation(models, inputs, outputs, test_inputs, test_outputs, loss_f, eval_f, epochs, args):
    ''' K-fold Cross Validation'''
    partitions = partition_data(inputs.size(0), args.partitions)

    # Saved for resetting later
    bs = args.batch_size

    # Holds training and eval loss curves for all models across all epochs
    cv_losses = torch.zeros(args.partitions, args.epochs, 3)

    # One model per data fold/partition, one optimizer per model
    fold_models = models
    fold_opts = [optim.Adam(model.parameters(), lr=args.learning_rate) for model in fold_models]
    #'''
    for i_fold in range(args.partitions):
        # Reset batch size for current model
        args.batch_size = bs
        model = fold_models[i_fold]
        opt = fold_opts[i_fold]

        # Split inputs into training and eval (nearly along cell lines)
        partition = partitions[i_fold]

        print('Fold %d:' % (i_fold+1))
        model, losses = train_model(model, inputs, outputs, partition, loss_f, eval_f, opt, epochs[i_fold], args, 'model_%d' % (i_fold+1))
        args.batch_size = bs

        # Put model back into list, probably not necessary but it's hard to know
        # when python passes by value or reference
        fold_models[i_fold] = model
        cv_losses[i_fold] = losses
        try:
            save_losses(cv_losses[:i_fold+1].cpu(), args.model_path+'/losses')
        except RuntimeError:
            pass
    #'''
    print('Fold-models trained, readying classification data.')
    args.partition_training = False
    partition = [slice(inputs.size(0)), slice(inputs.size(0), None), slice(0)]
    train_inputs = torch.cat([inputs, test_inputs], dim=0)
    true_outputs = torch.cat([outputs, test_outputs], dim=0)
    with torch.no_grad():
        fold_models[:-1] = [model.eval() for model in fold_models[:-1]]
        train_outputs = torch.zeros(train_inputs.size(0), args.partitions)

        n_batches = (train_inputs.size(0) // args.batch_size)+1
        # Batch the eval data
        for i_batch in range(n_batches):
            inds = slice(i_batch*args.batch_size, (i_batch+1)*args.batch_size)
            batch_inputs = train_inputs[inds].cuda()
            batch_outputs = true_outputs[inds].cuda()

            # Same reasoning as training: sometimes encounter 0-size batches
            if batch_inputs.size(0) == 0:
                continue

            # Build a sum of evaluation losses to average over later
            train_outputs[inds] = torch.stack([loss_f(model(batch_inputs, eval=True).view(-1), batch_outputs.view(-1)) for model in fold_models[:-1]], dim=1)

        # Convert to labels
        #train_outputs = train_outputs * -1 * 10
        train_outputs = train_outputs.argmin(dim=1)
        #train_outputs = (train_outputs < 0.1).float()
        print(train_outputs.mean(dim=1))

    model = fold_models[-1]
    opt = fold_opts[-1]
    print('Training Classifier:')
    model, losses = train_model(model, train_inputs, train_outputs, partition, nn.CrossEntropyLoss(reduction='none'), nn.CrossEntropyLoss(reduction='sum'), opt, epochs[-1], args, 'classifier')#nn.CrossEntropyLoss(reduction='none'), nn.CrossEntropyLoss(reduction='sum'), opt, epochs[-1], args, 'classifier')
    try:
        save_losses(losses.unsqueeze(0).cpu(), args.model_path+'/classifier_losses')
    except RuntimeError:
        pass

    fold_models[-1] = model

    loss_f = nn.BCELoss(reduction='none')
    ''' Ensemble and Individual Evaluation on Full Train Data '''
    with torch.no_grad():
        n_batches = (inputs.size(0) // args.batch_size)+1
        sum_loss = [0]*(args.partitions+1)

        for i_batch in range(n_batches):
            batch_inputs = inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
            batch_outputs = outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
            if batch_inputs.size(0) == 0:
                continue

            # Sum loss over each model and the mean of the models (ensemble)
            predictions = torch.cat([model(batch_inputs, eval=True).view(-1, 1) for model in fold_models[:-1]], dim=1)
            sum_loss[:-1] = [sum_loss[i] + eval_f(predictions[:,i], batch_outputs).item() for i in range(args.partitions)]
            weights = F.softmax(fold_models[-1](batch_inputs), dim=1)
            predictions = torch.sum(predictions * weights, dim=1)
            sum_loss[-1] += eval_f(predictions, batch_outputs).item()

        mean_loss = [sum_loss[i] / inputs.size(0) for i in range(args.partitions+1)]
        print(('Loss Of Ensemble Over All Folds at %d Epochs: %s' % (args.epochs, str(mean_loss)))+' '*10)

    ''' Ensemble and Individual Evaluation on Test Data '''
    with torch.no_grad():
        n_batches = (test_inputs.size(0) // args.batch_size)+1
        sum_loss = [0]*(args.partitions+1)

        for i_batch in range(n_batches):
            batch_inputs = test_inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
            batch_outputs = test_outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
            if batch_inputs.size(0) == 0:
                continue

            # Sum loss over each model and the mean of the models (ensemble)
            predictions = torch.cat([model(batch_inputs, eval=True).view(-1, 1) for model in fold_models[:-1]], dim=1)
            sum_loss[:-1] = [sum_loss[i] + eval_f(predictions[:,i], batch_outputs).item() for i in range(args.partitions)]
            weights = F.softmax(fold_models[-1](batch_inputs), dim=1)
            predictions = torch.sum(predictions * weights, dim=1)
            sum_loss[-1] += eval_f(predictions, batch_outputs).item()

        mean_loss = [sum_loss[i] / inputs.size(0) for i in range(args.partitions+1)]
        print(('Loss Of Ensemble Over Test Data at %d Epochs: %s' % (args.epochs, str(mean_loss)))+' '*10)

    return EnsembleModel(fold_models), cv_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a prediction over the eval set.')
    parser.add_argument('model_path', type=str, help='Relative path at which to save the model.')
    parser.add_argument('data_path', type=str, help='Relative path from which to load the data.')
    parser.add_argument('-partitions', type=int, default=1, help='Number of partitions for cross-fold validation.')
    parser.add_argument('-batch_size', type=int, default=256, help='Number of samples per batch.')
    parser.add_argument('-batch_size_annealing', type=float, default=1.0, help='Per-batch multiplier on batch size.')
    parser.add_argument('-epochs', type=int, default=25, help='Number of epochs to train for.')
    parser.add_argument('-learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('-loss_sampling', default=False, action='store_true', dest='loss_sampling', help='Flag to use loss-based sampling in place of traditional batching.')
    parser.add_argument('-save_rate', type=int, default=100, help='Save model and loss curves every ___ epochs.')
    parser.add_argument('-factored_model', default=False, action='store_true', dest='factored_model', help='Flag to use factored regression model.')
    parser.add_argument('-partition_training', default=False, action='store_true', dest='partition_training', help='Flag to train on the partition rather than eval on it.')

    args = parser.parse_args()
    if args.partitions == 1:
        args.partition_training = True
    if args.save_rate > args.epochs:
        args.save_rate = args.epochs

    try:
        args = pickle.load(open(args.model_path + '/args.pkl', 'rb'))
    except:
        print(args)


    # Load dataset
    train_inputs = torch.load(args.data_path + '/train_in.pt')
    train_outputs = torch.load(args.data_path + '/train_out.pt')
    test_inputs = torch.load(args.data_path + '/test_in.pt')
    test_outputs = torch.load(args.data_path + '/test_out.pt')

    # Load or create model
    models = build_models(args, train_inputs)
    epochs = [0]*len(models)
    for i in range(args.partitions):
        try:
            path = glob(args.model_path + '/model_%d_*.ptm' % (i+1))[0]
            epochs[i] = int(path[path.rfind('_')+1:path.rfind('.')])
            models[i] = torch.load(path).cuda()
        except:
            pass
    try:
        path = glob(args.model_path + '/classifier_*.ptm' % (i+1))[0]
        epochs[-1] = int(path[path.rfind('_')+1:path.rfind('.')])
        models[-1] = torch.load(path).cuda()
    except:
        pass
    epochs = [max(0, args.epochs - epoch) for epoch in epochs]
    print(models[0])
    print(args)

    # Create output directory if it doesn't exist
    os.makedirs(args.model_path, exist_ok=True)
    pickle.dump(args, open(args.model_path + '/args.pkl', 'wb'))

    print('Data and Model loaded, beginning training...')

    # Train model on dataset
    model, loss = train(models, train_inputs, train_outputs, test_inputs, test_outputs, epochs, args)

    # Save loss curves
    try:
        save_losses(loss, args.model_path+'/loss_curves')
    except RuntimeError:
        pass

    # Save model
    torch.save(model.cpu(), args.model_path + '/model.ptm')
