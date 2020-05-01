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

region_size = 8400#8400

def get_sequences(sequences, rand=False):
    if rand:
        offset = np.random.randint(1, 2+10000-region_size)
    else:
        offset = 5001-region_size//2
    return sequences[:, offset:offset+region_size].cuda()


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
def normpdf(shape):
    with torch.no_grad():
        x = torch.arange(shape[-1]).float()
        for dim in shape[:-1]:
            x = x.unsqueeze(0)
        x = x.expand(shape)
        x = x - torch.mean(x)
        x = x * 10 / region_size
        x = torch.exp(-(x**2)/2) / torch.sqrt(torch.FloatTensor([2*np.pi]))
        return x.cuda()

def step(model, inputs, outputs, loss_f, opt, n_layers):
    weighting = normpdf(outputs.shape)[outputs < 4]
    opt.zero_grad()
    predictions, r = model(inputs, n_layers)
    predictions = predictions.permute(0, 2, 1)[outputs < 4]
    loss = loss_f(predictions, outputs[outputs < 4]) * weighting
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


#@profile
def train_model(model, inputs, outputs, partition, loss_f, eval_f, opt, epochs, args, model_name='model'):
    model.cuda()
    model_path = args.model_path + '/' + model_name

    train_inputs = inputs[partition[0]]
    train_outputs = outputs[partition[0]]
    eval_inputs = inputs[partition[1]]
    eval_outputs = outputs[partition[1]]

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
    #try:
    #    data_mean = torch.mean(eval_outputs).detach()
    #    data_error = eval_f(eval_outputs.detach(), torch.ones(eval_outputs.size(0))*data_mean).detach() / eval_outputs.size(0)
    #except:
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
                batch_inputs = get_sequences(batch_inputs, rand=True)
                batch_outputs = batch_inputs
                '''
                # Build batches
                batch_indices = slice(i_batch*args.batch_size, (i_batch+1)*args.batch_size)
                batch_inputs = train_inputs[i_shuffle[batch_indices]].cuda()
                batch_outputs = train_outputs[i_shuffle[batch_indices]].cuda()
                '''

                # If the last batch is size 0, just skip it
                if batch_outputs.size(0) == 0:
                    continue

                # Perform gradient update on batch
                batch_losses = step(model, batch_inputs, batch_outputs, loss_f, opt, args.n_layers)
                train_losses += torch.sum(batch_losses).detach().cpu().item()
            train_losses = train_losses / train_inputs.size(0)

            # Set model to evaluation mode (turn off dropout and stuff)
            model.eval()

            n_batches_eval = min((eval_inputs.size(0) // args.batch_size), 10)
            sum_loss = 0

            # Batch the eval data
            #eval_inputs, eval_outputs = shuffle_data(eval_inputs, eval_outputs)
            i_shuffle = shuffle_data(eval_inputs, eval_outputs)
            for i_batch in range(n_batches_eval):
                batch_indices = slice(i_batch*args.batch_size,(i_batch+1)*args.batch_size)
                batch_inputs = eval_inputs[i_shuffle[batch_indices]]
                batch_inputs = get_sequences(batch_inputs, rand=True)
                batch_outputs = batch_inputs

                # Same reasoning as training: sometimes encounter 0-size batches
                if batch_outputs.size(0) == 0:
                    continue

                # Don't need to track operations/gradients for evaluation
                with torch.no_grad():
                    # Build a sum of evaluation losses to average over later
                    predictions, _ = model(batch_inputs, args.n_layers)
                    predictions = predictions.permute(0, 2, 1)[batch_outputs < 4]
                    weighting = normpdf(batch_outputs.shape)[batch_outputs < 4]
                    sum_loss += torch.sum(eval_f(predictions.squeeze(), batch_outputs[batch_outputs < 4].squeeze()) * weighting).item()

            n_batches_trainval = min((trainval_inputs.size(0) // args.batch_size), 10)
            sum_loss2 = 0
            # Batch the eval data
            #trainval_inputs, trainval_outputs = shuffle_data(trainval_inputs, trainval_outputs)
            i_shuffle = shuffle_data(trainval_inputs, trainval_outputs)
            for i_batch in range(n_batches_trainval):
                batch_indices = slice(i_batch*args.batch_size,(i_batch+1)*args.batch_size)
                batch_inputs = trainval_inputs[i_shuffle[batch_indices]]
                batch_inputs = get_sequences(batch_inputs, rand=True)
                batch_outputs = batch_inputs

                # Same reasoning as training: sometimes encounter 0-size batches
                if batch_outputs.size(0) == 0:
                    continue

                # Don't need to track operations/gradients for evaluation
                with torch.no_grad():
                    # Build a sum of evaluation losses to average over later
                    predictions, _ = model(batch_inputs, args.n_layers)
                    predictions = predictions.permute(0, 2, 1)[batch_outputs < 4]
                    weighting = normpdf(batch_outputs.shape)[batch_outputs < 4]
                    sum_loss2 += torch.sum(eval_f(predictions.squeeze(), batch_outputs[batch_outputs < 4].squeeze()) * weighting).item()

            # Calculate and print mean train and eval loss over the epoch
            mean_loss = sum_loss / (args.batch_size * n_batches_eval + 1)#eval_inputs.size(0)#
            mean_loss2 = sum_loss2 / (args.batch_size * n_batches_trainval + 1)#trainval_inputs.size(0)#
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


def train(model, inputs, outputs, epochs, args):
    # Separate loss functions for training and evaluation, allows for
    # potentially more flexible evaluation metrics
    loss_f = nn.CrossEntropyLoss(reduction='none')
    eval_f = nn.CrossEntropyLoss(reduction='none')

    partition = [slice(16000), slice(16000, None)]
    model, opt = build_opt(model)
    model, losses = train_model(model, inputs, outputs, partition, loss_f, eval_f, opt, epochs, args)
    losses = save_losses(losses.cpu().unsqueeze(0), args.model_path+'/losses')
    if isinstance(model, Exception):
        raise model
    return model, losses.cpu()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a prediction over the eval set.')
    parser.add_argument('model_path', type=str, help='Relative path at which to save the model.')
    parser.add_argument('data_path', type=str, help='Relative path from which to load the data.')
    parser.add_argument('-batch_size', type=int, default=0, help='Number of samples per batch.')
    parser.add_argument('-batch_size_annealing', type=float, default=-1, help='Per-batch multiplier on batch size.')
    parser.add_argument('-epochs', type=int, default=0, help='Number of epochs to train for.')
    parser.add_argument('-learning_rate', type=float, default=0, help='Learning rate.')
    parser.add_argument('-save_rate', type=int, default=100, help='Save model and loss curves every ___ epochs.')
    parser.add_argument('-n_layers', type=int, default=-1, help='Num Layers to train.')

    args = parser.parse_args()

    try:
        # Epochs, Batch Size, and Batch Size Annealing can be optionally changed on subsequent runs
        epochs = args.epochs
        batch_size = args.batch_size
        batch_size_annealing = args.batch_size_annealing
        learning_rate = args.learning_rate
        n_layers = args.n_layers
        args = pickle.load(open(args.model_path + '/args.pkl', 'rb'))
        if epochs > 0:
            args.epochs = epochs
        if batch_size > 0:
            args.batch_size = batch_size
        if batch_size_annealing >= 0:
            args.batch_size_annealing = batch_size_annealing
        if learning_rate > 0:
            args.learning_rate = learning_rate
        if n_layers >= 0:
            args.n_layers = n_layers
    except:
        # True Defaults
        if epochs == 0:
            args.epochs = 1000
        if batch_size == 0:
            args.batch_size = 256
        if batch_size_annealing < 0:
            args.batch_size_annealing = 0.0
        if learning_rate <= 0:
            args.learning_rate = 1e-4
        if n_layers < 0:
            args.n_layers = 100


    if args.save_rate > args.epochs:
        args.save_rate = args.epochs

    # Load dataset
    print('Loading data...')
    inputs = torch.load(args.data_path + '/sequences.pt').long()
    outputs = inputs
    print('Data loaded.')

    # Load or create model
    model = SeqModel()
    epochs, model2 = load_model(args.model_path + '/model_*.ptm')
    epochs = args.epochs - epochs
    if model2 is not None:
        model = model2
    if isinstance(model, tuple):
        print(model[0])
    else:
        print(model)
    print(args)

    # Create output directory if it doesn't exist
    os.makedirs(args.model_path, exist_ok=True)
    pickle.dump(args, open(args.model_path + '/args.pkl', 'wb'))

    print('Data and Model loaded, beginning training...')

    # Train model on dataset
    model, loss = train(model, inputs, outputs, epochs, args)
