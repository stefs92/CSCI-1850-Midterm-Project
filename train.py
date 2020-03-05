import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import traceback
from model import *


def save_losses(losses, destination):
    colors = ['red', 'blue', 'yellow', 'black', 'green', 'magenta', 'cyan']
    markers = ['o', 's', '*', '+', 'D', '|', '_']
    losses = losses.view(losses.size(0), losses.size(1), -1)
    torch.save(losses.detach(), destination+'.pt')
    losses = losses.numpy()

    for i in range(losses.shape[0]):
        plt.plot(range(losses.shape[1]), losses[i, :, 0], color=colors[i], marker=markers[i], linestyle='solid')
        if losses.shape[2] == 2:
            plt.plot(range(losses.shape[1]), losses[i, :, 1], color=colors[i], marker=markers[i], linestyle='dashed')

    plt.savefig(destination+'.png')
    plt.clf()


def build_model(args):
    return ConvolutionalModel().cuda()


def shuffle_data(*tensors):
    i_shuffle = torch.randperm(tensors[0].size(0))
    return [tensors[i][i_shuffle] for i in range(len(tensors))]

'''
Performs a single gradient update and returns losses
'''
def step(model, inputs, outputs, loss_f, opt):
    predictions = model(inputs).view(-1)
    loss = loss_f(predictions, outputs)
    torch.mean(loss).backward()
    opt.step()
    opt.zero_grad()
    return loss


def train(model, inputs, outputs, test_inputs, test_outputs, args):
    # Saved for resetting later
    bs = args.batch_size

    data_size = inputs.size(0)
    partition_size = data_size // args.partitions

    # Separate loss functions for training and evaluation, allows for
    # potentially more flexible evaluation metrics
    loss_f = nn.MSELoss(reduction='none')
    eval_f = nn.MSELoss(reduction='sum')

    ''' K-fold Cross Validation'''
    # Holds training and eval loss curves for all models across all epochs
    cv_losses = torch.zeros(args.partitions, args.epochs, 2)

    # One model per data fold/partition, one optimizer per model
    fold_models = [copy.deepcopy(model) for i in range(args.partitions)]
    fold_opts = [optim.Adam(model.parameters(), lr=args.learning_rate) for model in fold_models]

    # Initialized here for scoping
    mean_loss = 0
    for i_fold in range(args.partitions):
        # Reset batch size for current model
        args.batch_size = bs
        model = fold_models[i_fold]
        opt = fold_opts[i_fold]

        # Split inputs into training and eval (nearly along cell lines)
        eval_inputs = inputs[i_fold*partition_size:(i_fold+1)*partition_size]
        eval_outputs = outputs[i_fold*partition_size:(i_fold+1)*partition_size]
        train_inputs = torch.cat([inputs[:i_fold*partition_size], inputs[(i_fold+1)*partition_size:]])
        train_outputs = torch.cat([outputs[:i_fold*partition_size], outputs[(i_fold+1)*partition_size:]])

        # Pre-calculate the mean of the eval data and from it the error for R^2
        data_mean = torch.mean(eval_outputs).detach()
        data_error = eval_f(eval_outputs.detach(), torch.ones(eval_outputs.size(0))*data_mean).detach() / eval_outputs.size(0)

        n_batches = (train_inputs.size(0) // args.batch_size)+1
        train_losses = torch.zeros(n_batches)

        # Initialize logit tensor for loss-based importance sampling
        if args.loss_sampling:
            logits = torch.zeros(train_inputs.size(0))

        # Run train/eval loop over specified number of epochs
        for i_epoch in range(args.epochs):
            # Increase batch size according to specified schedule
            args.batch_size = int(args.batch_size * args.batch_size_annealing)

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
                train_losses[i_batch] = torch.mean(batch_losses).detach()

                # If using loss sampling, update logits
                if args.loss_sampling:
                    with torch.no_grad():
                        logits[batch_indices] = batch_losses.cpu()

            # Don't need to track operations/gradients for evaluation
            with torch.no_grad():
                # Set model to evaluation mode (turn off dropout and stuff)
                model.eval()

                n_batches = (eval_inputs.size(0) // args.batch_size)+1
                sum_loss = 0

                # Batch the eval data
                for i_batch in range(n_batches):
                    batch_inputs = eval_inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
                    batch_outputs = eval_outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()

                    # Same reasoning as training: sometimes encounter 0-size batches
                    if batch_inputs.size(0) == 0:
                        continue

                    # Build a sum of evaluation losses to average over later
                    predictions = model(batch_inputs).view(-1)
                    sum_loss += eval_f(predictions, batch_outputs).item()

                # Calculate and print mean train and eval loss over the epoch
                mean_loss = sum_loss / eval_inputs.size(0)
                cv_losses[i_fold, i_epoch, 0] = torch.mean(train_losses)
                cv_losses[i_fold, i_epoch, 1] = mean_loss
                print('Fold %d, Epoch %d Mean Train / Eval Loss and R^2 Value: %.3f / %.3f / %.3f ' % (i_fold+1, i_epoch+1, cv_losses[i_fold, i_epoch, 0], cv_losses[i_fold, i_epoch, 1], 1 - mean_loss / data_error), end='\r')

        # Put model back into list, probably not necessary but it's hard to know
        # when python passes by value or reference
        fold_models[i_fold] = model
        print('') # to keep only the final epoch losses from each fold

    # If args.epochs==0 then there's no value in performing evaluation
    if args.epochs > 0:
        ''' Ensemble and Individual Evaluation on All Non-Test Data '''
        with torch.no_grad():
            n_batches = (inputs.size(0) // args.batch_size)+1
            sum_loss = [0]*(args.partitions+1)

            for i_batch in range(n_batches):
                batch_inputs = inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
                batch_outputs = outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
                if batch_inputs.size(0) == 0:
                    continue

                # Sum loss over each model and the mean of the models (ensemble)
                predictions = torch.cat([model(batch_inputs).view(-1, 1) for model in fold_models], dim=1)
                sum_loss[:-1] = [sum_loss[i] + eval_f(predictions[:,i], batch_outputs).item() for i in range(args.partitions)]
                predictions = torch.mean(predictions, dim=1).view(-1)
                sum_loss[-1] += eval_f(predictions, batch_outputs).item()

            mean_loss = [sum_loss[i] / inputs.size(0) for i in range(args.partitions+1)]
            print(('Loss Of Ensemble Over All Folds at %d Epochs: %s' % (args.epochs, str(mean_loss)))+' '*10)


    ''' Training on All Data '''
    # Holds training and test loss curves for all models across epochs
    ad_losses = torch.zeros(args.partitions, 1, 2)

    # See k-fold cross validation for comments on the specifics of all this stuff
    train_inputs = inputs
    train_outputs = outputs
    data_mean = torch.mean(test_outputs).detach()
    data_error = eval_f(test_outputs.detach(), torch.ones(test_outputs.size(0))*data_mean).detach() / test_outputs.size(0)

    n_batches = (train_inputs.size(0) // args.batch_size)+1
    train_losses = torch.zeros(args.partitions, n_batches)

    if args.loss_sampling:
        logits = torch.zeros(inputs.size(0))
    i_epoch = 0
    try:
        while True:
            i_epoch += 1

            # Train Epoch
            [model.train() for model in fold_models]
            n_batches = (train_inputs.size(0) // args.batch_size)+1
            train_inputs, train_outputs = shuffle_data(train_inputs, train_outputs)
            for i_batch in range(n_batches):
                if args.loss_sampling and i_epoch > 0:
                    batch_indices = torch.multinomial(logits, args.batch_size, replacement=True)
                else:
                    batch_indices = slice(i_batch*args.batch_size, (i_batch+1)*args.batch_size)

                batch_inputs = train_inputs[batch_indices].cuda()
                batch_outputs = train_outputs[batch_indices].cuda()
                if batch_inputs.size(0) == 0:
                    continue
                for i_model in range(args.partitions):
                    batch_losses = step(fold_models[i_model], batch_inputs, batch_outputs, loss_f, fold_opts[i_model])
                    train_losses[i_model, i_batch] = torch.sum(batch_losses).detach()

                if args.loss_sampling:
                    with torch.no_grad():
                        logits[batch_indices] = batch_losses.cpu()

            # Eval Epoch
            with torch.no_grad():
                [model.eval() for model in fold_models]
                n_batches = (test_inputs.size(0) // args.batch_size)+1
                sum_loss = [0]*args.partitions

                for i_batch in range(n_batches):
                    batch_inputs = test_inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
                    batch_outputs = test_outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
                    if batch_inputs.size(0) == 0:
                        continue
                    for i_model in range(args.partitions):
                        predictions = fold_models[i_model](batch_inputs).view(-1)
                        sum_loss[i_model] += eval_f(predictions, batch_outputs).item()

                mean_loss = [sum_loss[i] / test_inputs.size(0) for i in range(args.partitions)]
                ad_losses = torch.cat([ad_losses, torch.zeros(args.partitions, 1, 2).to(ad_losses)], dim=1)
                ad_losses[:, -1, 0] = torch.sum(train_losses, dim=1) / train_inputs.size(0)
                ad_losses[:, -1, 1] = torch.FloatTensor(mean_loss).to(ad_losses)
                print('All Data Epoch %d Mean Train Loss, Delta Test Loss, and R^2 Value: %s / %s / %s ' % (i_epoch, str(ad_losses[:, i_epoch, 0].cpu().numpy()), str((ad_losses[:, 1, 1] - ad_losses[:, i_epoch, 1]).cpu().numpy()), str([1 - mean_loss[i] / data_error for i in range(args.partitions)])), end='\r')
    except Exception:
        traceback.print_exc()
    finally:
        return EnsembleModel(fold_models), cv_losses.cpu(), ad_losses[:, 1:].cpu()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a prediction over the eval set.')
    parser.add_argument('model_path', type=str, help='Relative path at which to save the model.')
    parser.add_argument('-partitions', type=int, default=3, help='Number of partitions for cross-fold validation.')
    parser.add_argument('-batch_size', type=int, default=256, help='Number of samples per batch.')
    parser.add_argument('-batch_size_annealing', type=float, default=1.0, help='Per-batch multiplier on batch size.')
    parser.add_argument('-epochs', type=int, default=25, help='Number of epochs to train for.')
    parser.add_argument('-learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('-loss_sampling', default=False, action='store_true', dest='loss_sampling', help='Flag to use loss-based sampling in place of traditional batching.')

    args = parser.parse_args()

    # Load or create model
    try:
        model = torch.load(args.model_path + '/model.ptm').cuda()
    except:
        model = build_model(args)

    # Load dataset
    train_inputs = torch.load('resampled_in.pt')
    train_outputs = torch.load('resampled_out.pt')
    test_inputs = torch.load('test_in.pt')
    test_outputs = torch.load('test_out.pt')

    # Train model on dataset
    model, cvloss, adloss = train(model, train_inputs, train_outputs, test_inputs, test_outputs, args)

    # Create output directory if it doesn't exist
    os.makedirs(args.model_path, exist_ok=True)

    # Save loss curves
    try:
        save_losses(cvloss, args.model_path+'/cross_validation_losses')
    except RuntimeError:
        pass
    save_losses(adloss, args.model_path+'/all_data_losses')

    # Save model
    torch.save(model.cpu(), args.model_path + '/model.ptm')
