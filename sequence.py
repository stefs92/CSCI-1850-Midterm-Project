# Example submission of predicted values
import pandas as pd
from info import eval_cells
import argparse
import torch
#from train import *
import numpy as np
from train_seq import get_sequences, region_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a prediction over the eval set.')
    parser.add_argument('model_path', type=str, help='Relative path to the model used to generate predictions.')
    parser.add_argument('data_path', type=str, help='Relative path from which to load the data.')
    parser.add_argument('-batch_size', type=int, default=1024, help='Size of batches for model predictions.')

    args = parser.parse_args()

    model = torch.load(args.model_path)[0].cuda()
    model.eval()

    inputs = get_sequences(torch.load(args.data_path + '/sequences.pt'), rand=False).cuda()
    print(inputs.shape)
    num_batches = (inputs.size(0) // args.batch_size) + 1

    with torch.no_grad():
        predictions = [model(inputs[args.batch_size*i:args.batch_size*(i+1)], n_layers=30)[1].cpu().squeeze() for i in range(num_batches)]
        predictions = torch.cat(predictions)
        print(predictions.shape)
        submission_title = args.model_path[args.model_path.find('/')+1:args.model_path.rfind('/')]
        torch.save(predictions, args.data_path + '/predicted_' + submission_title + '.pt')

        mean = predictions.mean()
        var = predictions.std()

        print('Submission mean: %.3f' % mean)
        print('Submission standard deviation: %.3f' % var)
