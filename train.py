import argparse
import copy
import os
from glob import glob
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import matplotlib.pyplot as plt
import traceback
from itertools import accumulate as accumulate
from model import *
from loss import *
#from memory_profiler import profile

region_size = 8300

def get_sequences(labels, sequences):
    indices = np.nonzero(labels.numpy() == sequences[:, :1].numpy())[0]
    return sequences[indices, 5001-region_size//2:5001+region_size//2].cuda()


def save_model(model, opt, path):
    model.cpu()
    opt.zero_grad()
    torch.save((model, opt), path)
    model.cuda()


def save_losses(losses, destination):
    try:
        old_losses = torch.load(glob(destination+'.pt')[0])[:losses.size(0), :losses.size(1)]
    except:
        old_losses = torch.zeros_like(losses).to(losses)
    indices = losses[:old_losses.size(0),:old_losses.size(1)] == 0
    losses[:old_losses.size(0),:old_losses.size(1)][indices] = old_losses[:,:][indices]
    loss_len = torch.sum(torch.sum(losses[:,:,0] != 0, dim=0) != 0)
    losses = losses[:,:loss_len]
    old_losses = old_losses[:,:loss_len]

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
    lossesnpy = losses.numpy()

    for i in range(losses.shape[0]):
        plt.plot(range(losses.shape[1]), lossesnpy[i, :, 0], color=colors[i], marker=markers[i], linestyle='solid')
        if losses.shape[2] > 1:
            plt.plot(range(losses.shape[1]), lossesnpy[i, :, 1], color=colors[i], marker=markers[i], linestyle='dashed')
        if losses.shape[2] > 2:
            plt.plot(range(losses.shape[1]), lossesnpy[i, :, 2], color=colors[i], marker=markers[i], linestyle='dotted')

    plt.savefig(destination+'.png')
    plt.clf()

    return losses


def build_models(args, input_size):
    class_ = ConvolutionalModel

    if args.partitions > 1:
        weightor = [ClassificationModel(in_size=input_size, n_classes=args.partitions)]
    else:
        weightor = []
    predictors = []
    for partition in range(args.partitions):
        predictors += [class_(in_size=input_size, activation=None)]#[MixModel()]#
    return predictors + weightor


def partition_data(dsize, n_partitions):
    partition_size = dsize // n_partitions
    partition_sizes = [partition_size] * (n_partitions - 1) + [dsize - (partition_size * (n_partitions - 1))]
    partition_edges = [0] + list(accumulate(partition_sizes))
    partition_slices = [(slice(0, partition_edges[i]), slice(partition_edges[i], partition_edges[i+1]), slice(partition_edges[i+1], None)) for i in range(n_partitions)]
    return partition_slices


def shuffle_data(*tensors):
    i_shuffle = torch.randperm(tensors[0].size(0))
    return i_shuffle

'''
Performs a single gradient update and returns losses
'''
def step(model, inputs, outputs, loss_f, opt):
    opt.zero_grad()
    predictions = model(inputs)#[0], inputs[1])
    loss = loss_f(predictions.squeeze(), outputs.squeeze())
    torch.mean(loss).backward()
    opt.step()
    opt.zero_grad()
    return loss


def del_checkpoints(paths):
    for path in paths:
        os.remove(path)


def load_model(path):
    try:
        paths = glob(path)
        path_epochs = [int(path[path.rfind('_')+1:path.rfind('.')]) for path in paths]
        max_epochs = 0
        amax_epochs = 0
        for j in range(len(paths)):
            if path_epochs[j] > max_epochs:
                max_epochs = path_epochs[j]
                amax_epochs = j
        path = paths[amax_epochs]
        del_checkpoints(paths[:amax_epochs] + paths[amax_epochs+1:])
        model, opt = build_opt(torch.load(path))
        return max_epochs, (model.cpu(), opt)
    except:
        return 0, None


def build_opt(model):
    if isinstance(model, tuple):
        opt = model[1]
        model = model[0]
    else:
        opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    return model, opt



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

#@profile
def train_model(model, inputs, outputs, partition, loss_f, eval_f, opt, epochs, args, model_name='model'):
    model.cuda()
    model_path = args.model_path + '/' + model_name

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

    train_data = TensorDataset(train_inputs, train_outputs)

    # Resample train set to a normal distribution over targets
    #if isinstance(outputs, torch.FloatTensor):
    #    resample_probs = build_resample_probs(train_outputs)
    #    resample_indices = torch.multinomial(resample_probs, train_outputs.size(0), replacement=True)
    trainval_inputs = train_inputs#copy.deepcopy(train_inputs)
    trainval_outputs = train_outputs#copy.deepcopy(train_outputs)
    #if isinstance(outputs, torch.FloatTensor):
    #    train_inputs = train_inputs[resample_indices]
    #    train_outputs = train_outputs[resample_indices]

    # Pre-calculate the mean of the eval data and from it the error for R^2
    try:
        data_mean = torch.mean(eval_outputs).detach()
        data_error = eval_f(eval_outputs.detach(), torch.ones(eval_outputs.size(0))*data_mean).detach() / eval_outputs.size(0)
    except:
        #data_mean = torch.mode(eval_outputs, dim=0).values.view(-1).detach()
        data_error = torch.FloatTensor([1]).to(eval_inputs)

    train_losses = 0
    losses = torch.zeros(args.epochs, 3)

    try:
        # Run train/eval loop over specified number of epochs
        for i_epoch in range(args.epochs):
            # Increase batch size according to specified schedule
            args.batch_size = int(args.batch_size + args.batch_size_annealing)
            if i_epoch < (args.epochs - epochs):
                continue

            # Prep Data Loader
            train_loader = DataLoader(train_data,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      pin_memory=True)

            # Set model to training mode
            model.train()

            i_shuffle = shuffle_data(train_inputs, train_outputs)

            # Batch training data
            for i_batch, batch in enumerate(train_loader):
                batch_inputs, batch_outputs = batch
                indices = torch.cat([torch.from_numpy(np.nonzero(args.sequences[0] == batch_inputs[i,0,0].numpy())[0]) for i in range(batch_inputs.shape[0])])
                #seq_inputs = args.sequences[1][indices].cuda()
                batch_inputs = torch.cat([batch_inputs[:,1:,:], args.sequences[1][indices]], dim=1).cuda()
                #batch_inputs = batch_inputs[:,1:,:].cuda()
                batch_outputs = batch_outputs.cuda()

                # Perform gradient update on batch
                batch_losses = step(model, batch_inputs, batch_outputs, loss_f, opt)
                #batch_losses = step(model, (batch_inputs, seq_inputs), batch_outputs, loss_f, opt)
                train_losses += torch.sum(batch_losses).detach().cpu().item()
            train_losses = train_losses / train_inputs.size(0)

            # Set model to evaluation mode (turn off dropout and stuff)
            model.eval()

            n_batches_eval =  min((eval_inputs.size(0) // args.batch_size), 10)
            sum_loss = 0

            # Batch the eval data
            #eval_inputs, eval_outputs = shuffle_data(eval_inputs, eval_outputs)
            i_shuffle = shuffle_data(eval_inputs, eval_outputs)
            for i_batch in range(n_batches_eval):
                batch_indices = slice(i_batch*args.batch_size,(i_batch+1)*args.batch_size)
                batch_inputs = eval_inputs[i_shuffle[batch_indices]]
                indices = torch.cat([torch.from_numpy(np.nonzero(args.sequences[0] == batch_inputs[i,0,0].numpy())[0]) for i in range(batch_inputs.shape[0])])
                #seq_inputs = args.sequences[1][indices].cuda()
                batch_inputs = torch.cat([batch_inputs[:,1:,:], args.sequences[1][indices]], dim=1).cuda()
                #batch_inputs = batch_inputs[:,1:,:].cuda()
                batch_outputs = eval_outputs[i_shuffle[batch_indices]].cuda()

                # Same reasoning as training: sometimes encounter 0-size batches
                if batch_outputs.size(0) == 0:
                    continue

                # Don't need to track operations/gradients for evaluation
                with torch.no_grad():
                    # Build a sum of evaluation losses to average over later
                    predictions = model(batch_inputs, eval=True).squeeze()
                    #predictions = model(batch_inputs, seq_inputs, eval=True).squeeze()#torch.sigmoid(model(batch_inputs, eval=True).squeeze())
                    sum_loss += eval_f(predictions.squeeze(), batch_outputs.squeeze()).item()

            n_batches_trainval = min((trainval_inputs.size(0) // args.batch_size), 10)
            sum_loss2 = 0
            # Batch the eval data
            #trainval_inputs, trainval_outputs = shuffle_data(trainval_inputs, trainval_outputs)
            i_shuffle = shuffle_data(trainval_inputs, trainval_outputs)
            for i_batch in range(n_batches_trainval):
                batch_indices = slice(i_batch*args.batch_size,(i_batch+1)*args.batch_size)
                batch_inputs = trainval_inputs[i_shuffle[batch_indices]]
                indices = torch.cat([torch.from_numpy(np.nonzero(args.sequences[0] == batch_inputs[i,0,0].numpy())[0]) for i in range(batch_inputs.shape[0])])
                #seq_inputs = args.sequences[1][indices].cuda()
                batch_inputs = torch.cat([batch_inputs[:,1:,:], args.sequences[1][indices]], dim=1).cuda()
                #batch_inputs = batch_inputs[:,1:,:].cuda()
                batch_outputs = trainval_outputs[i_shuffle[batch_indices]].cuda()

                # Same reasoning as training: sometimes encounter 0-size batches
                if batch_outputs.size(0) == 0:
                    continue

                # Don't need to track operations/gradients for evaluation
                with torch.no_grad():
                    # Build a sum of evaluation losses to average over later
                    predictions = model(batch_inputs, eval=True).squeeze()
                    #predictions = model(batch_inputs, seq_inputs, eval=True).squeeze()#torch.sigmoid(model(batch_inputs, eval=True).squeeze())
                    sum_loss2 += eval_f(predictions.squeeze(), batch_outputs.squeeze()).item()

            # Calculate and print mean train and eval loss over the epoch
            mean_loss = sum_loss / (args.batch_size * n_batches_eval)#eval_inputs.size(0)#
            mean_loss2 = sum_loss2 / (args.batch_size * n_batches_trainval)#trainval_inputs.size(0)#
            losses[i_epoch, 0] = train_losses
            losses[i_epoch, 1] = mean_loss
            losses[i_epoch, 2] = mean_loss2
            print('Epoch %d Mean Train / TrainVal / Eval Loss and R^2 Value: %.3f / %.3f / %.3f / %.3f ' % (i_epoch+1, losses[i_epoch, 0], losses[i_epoch, 2], losses[i_epoch, 1], 1 - (mean_loss / data_error).item()), end='\r')

            if (i_epoch+1) % args.save_rate == 0:
                save_model(model, opt, model_path + '_%d.ptm' % (i_epoch+1))

        print('') # to keep only the final epoch losses from each fold
        return model.cpu(), losses.cpu()
    except (Exception, KeyboardInterrupt) as e:
        save_model(model, opt, model_path + '_%d.ptm' % i_epoch)
        #raise e
        return e, losses.cpu()


def train(models, inputs, outputs, test_inputs, test_outputs, epochs, args):
    # Separate loss functions for training and evaluation, allows for
    # potentially more flexible evaluation metrics
    loss_f = nn.MSELoss(reduction='none')
    eval_f = nn.MSELoss(reduction='sum')

    if args.partitions > 1:
        print('Training an ensemble model via %d-fold cross-validation.' % args.partitions)
        return cross_validation(models, inputs, outputs, test_inputs, test_outputs, loss_f, eval_f, epochs, args)
    else:
        print('Training a single model on train and test data.')
        args.partition_training = False
        partition = [slice(train_inputs.size(0)), slice(train_inputs.size(0), None), slice(0)]
        inputs = torch.cat([train_inputs, test_inputs], dim=0)
        outputs = torch.cat([train_outputs, test_outputs], dim=0)
        model, opt = build_opt(models[0])
        model, losses = train_model(model, inputs, outputs, partition, loss_f, eval_f, opt, epochs[0], args)
        losses = save_losses(losses.cpu().unsqueeze(0), args.model_path+'/losses')
        if isinstance(model, Exception):
            raise model
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
    #'''
    for i_fold in range(args.partitions):
        # Reset batch size for current model
        args.batch_size = bs
        model = fold_models[i_fold]
        model, opt = build_opt(model)

        # Split inputs into training and eval (nearly along cell lines)
        partition = partitions[i_fold]

        print('Fold %d:' % (i_fold+1))
        model, losses = train_model(model, inputs, outputs, partition, loss_f, eval_f, opt, epochs[i_fold], args, 'model_%d' % (i_fold+1))
        args.batch_size = bs

        # Put model back into list, probably not necessary but it's hard to know
        # when python passes by value or reference
        cv_losses[i_fold] = losses
        try:
            cv_losses = save_losses(cv_losses.cpu(), args.model_path+'/losses')
        except RuntimeError:
            pass
        if isinstance(model, RuntimeError):
            print('')
            print('RuntimeError Encountered, Continuing...')
        elif isinstance(model, (Exception, KeyboardInterrupt)):
            raise model
        else:
            fold_models[i_fold] = model
    #'''
    print('Fold-models trained, readying classification data.')
    args.partition_training = False
    partition = [slice(inputs.size(0)), slice(inputs.size(0), None), slice(0)]
    train_inputs = torch.cat([inputs, test_inputs], dim=0)
    true_outputs = torch.cat([outputs, test_outputs], dim=0)
    with torch.no_grad():
        fold_models[:-1] = [model.eval().cuda() for model in fold_models[:-1]]
        train_outputs = torch.zeros(train_inputs.size(0), args.partitions)

        n_batches = (train_inputs.size(0) // args.batch_size)+1
        # Batch the eval data
        for i_batch in range(n_batches):
            inds = slice(i_batch*args.batch_size, (i_batch+1)*args.batch_size)
            batch_inputs = train_inputs[inds]

            if batch_inputs.size(0) == 0:
                continue

            indices = torch.cat([torch.from_numpy(np.nonzero(args.sequences[0] == batch_inputs[i,0,0].numpy())[0]) for i in range(batch_inputs.shape[0])])
            #seq_inputs = args.sequences[1][indices].cuda()
            batch_inputs = torch.cat([batch_inputs[:,1:,:], args.sequences[1][indices]], dim=1).cuda()
            batch_outputs = true_outputs[inds].cuda()

            # Same reasoning as training: sometimes encounter 0-size batches
            #if batch_inputs.size(0) == 0:
            #    continue

            # Build a sum of evaluation losses to average over later
            train_outputs[inds] = torch.stack([loss_f(model(batch_inputs, eval=True).view(-1), batch_outputs.view(-1)) for model in fold_models[:-1]], dim=1)

        # Convert to labels
        #train_outputs = train_outputs * -1 * 10
        train_outputs = train_outputs.argmin(dim=1)
        #train_outputs = (train_outputs < 0.1).float()
        #print(train_outputs.mean(dim=1))

    for model in fold_models[:-1]:
        model.cpu()
    model, opt = build_opt(fold_models[-1])
    print('Training Classifier:')
    model, losses = train_model(model, train_inputs, train_outputs, partition, nn.CrossEntropyLoss(reduction='none'), nn.CrossEntropyLoss(reduction='sum'), opt, epochs[-1], args, 'classifier')
    try:
        losses = save_losses(losses.unsqueeze(0).cpu(), args.model_path+'/classifier_losses')[0]
    except RuntimeError:
        pass

    if isinstance(model, (Exception, KeyboardInterrupt)):
        raise model

    fold_models[-1] = model
    for model in fold_models:
        model.cuda()

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

        mean_loss = [sum_loss[i] / test_inputs.size(0) for i in range(args.partitions+1)]
        print(('Loss Of Ensemble Over Test Data at %d Epochs: %s' % (args.epochs, str(mean_loss)))+' '*10)

    return EnsembleModel(fold_models), cv_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a prediction over the eval set.')
    parser.add_argument('model_path', type=str, help='Relative path at which to save the model.')
    parser.add_argument('data_path', type=str, help='Relative path from which to load the data.')
    parser.add_argument('-partitions', type=int, default=1, help='Number of partitions for cross-fold validation.')
    parser.add_argument('-batch_size', type=int, default=0, help='Number of samples per batch.')
    parser.add_argument('-batch_size_annealing', type=float, default=-1, help='Per-batch multiplier on batch size.')
    parser.add_argument('-epochs', type=int, default=0, help='Number of epochs to train for.')
    parser.add_argument('-learning_rate', type=float, default=0, help='Learning rate.')
    parser.add_argument('-save_rate', type=int, default=100, help='Save model and loss curves every ___ epochs.')
    parser.add_argument('-partition_training', default=False, action='store_true', dest='partition_training', help='Flag to train on the partition rather than eval on it.')
    parser.add_argument('-sequences', help='Placeholder, please ignore.')

    args = parser.parse_args()

    try:
        # Epochs, Batch Size, and Batch Size Annealing can be optionally changed on subsequent runs
        epochs = args.epochs
        batch_size = args.batch_size
        batch_size_annealing = args.batch_size_annealing
        learning_rate = args.learning_rate
        args = pickle.load(open(args.model_path + '/args.pkl', 'rb'))
        if epochs > 0:
            args.epochs = epochs
        if batch_size > 0:
            args.batch_size = batch_size
        if batch_size_annealing >= 0:
            args.batch_size_annealing = batch_size_annealing
        if learning_rate > 0:
            args.learning_rate = learning_rate
    except:
        # True Defaults
        if epochs == 0:
            args.epochs = 1000
        if batch_size == 0:
            args.batch_size = 256
        if batch_size_annealing < 0:
            args.batch_size_annealing = 0.0
        if learning_rate <= 0:
            args.learning_rate = 1e-3

        # Necessary Enforcement
        if args.partitions == 1:
            args.partition_training = True

    if args.save_rate > args.epochs:
        args.save_rate = args.epochs

    # Load dataset
    print('Loading data...')
    train_inputs = torch.load(args.data_path + '/train_in.pt')
    train_outputs = torch.load(args.data_path + '/train_out.pt')
    test_inputs = torch.load(args.data_path + '/test_in.pt')
    test_outputs = torch.load(args.data_path + '/test_out.pt')
    try:
        '''
        seq_paths = glob(args.data_path+'/predicted_*.pt')
        sequences = torch.load(seq_paths[0])
        input_size = train_inputs.shape[1] + sequences.shape[1] - 1
        seq_labels = torch.load(args.data_path + '/sequences.pt')[:,:1]
        sequences = (seq_labels, sequences)
        '''
        sequences = torch.load(args.data_path+'/sequences.pt').permute(0, 2, 1)
        input_size = train_inputs.shape[1] + sequences.shape[1] - 1
        seq_labels = torch.load(args.data_path + '/sequence_labels.pt').long().numpy()
        sequences = (seq_labels, sequences)
    except Exception as e:
        sequences = None
        input_size = train_inputs.shape[1] - 1
    print('Data loaded.')

    # Load or create model
    models = build_models(args, input_size)
    epochs = [0]*len(models)
    if args.partitions == 1:
        epochs[0], model = load_model(args.model_path + '/model_*.ptm')
        if model is not None:
            models[0] = model
    else:
        for i in range(args.partitions):
            epochs[i], model = load_model(args.model_path + '/model_%d_*.ptm' % (i+1))
            if model is not None:
                models[i] = model
        epochs[-1], model = load_model(args.model_path + '/classifier_*.ptm')
        if model is not None:
            models[-1] = model
    epochs = [max(0, args.epochs - epoch) for epoch in epochs]
    if isinstance(models[0], tuple):
        print(models[0][0])
    else:
        print(models[0])
    print(args)

    # Create output directory if it doesn't exist
    os.makedirs(args.model_path, exist_ok=True)
    pickle.dump(args, open(args.model_path + '/args.pkl', 'wb'))

    # Load sequences only after all of the args stuff is done
    #args.sequences = torch.load(args.data_path + '/sequences.pt')
    #args.sequences = (args.sequences[:,:1], args.sequences[:,1:])
    args.sequences = sequences

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
