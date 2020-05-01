# Example submission of predicted values
import pandas as pd
from info import eval_cells
import argparse
import torch
from glob import glob
#from train import *
import numpy as np


def submit(predictions, title):
    cell_list = []
    gene_list = []
    eval_data = np.load('eval.npz')
    for cell in eval_cells:
        cell_data = eval_data[cell]
        cell_list.extend([cell]*len(cell_data))
        genes = cell_data[:,0,0].astype('int32')
        gene_list.extend(genes)

    id_column = [] # ID is {cell type}_{gene id}
    for idx in range(len(cell_list)):
        id_column.append(f'{cell_list[idx]}_{gene_list[idx]}')

    df_data = {'id': id_column, 'expression' : predictions}
    submit_df = pd.DataFrame(data=df_data)

    submit_df.to_csv('%s.csv' % title, header=True, index=False, index_label=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a prediction over the eval set.')
    parser.add_argument('model_path', type=str, help='Relative path to the model used to generate predictions.')
    parser.add_argument('data_path', type=str, help='Relative path from which to load the data.')
    parser.add_argument('-batch_size', type=int, default=1024, help='Size of batches for model predictions.')

    args = parser.parse_args()

    model = torch.load(args.model_path)[0].cuda()
    model.eval()

    inputs = torch.load(args.data_path + '/submission_in.pt')
    try:
        seq_paths = glob(args.data_path+'/predicted_*.pt')
        sequences = torch.load(seq_paths[0])
        input_size = inputs.shape[1] + sequences.shape[1] - 1
        seq_labels = torch.load(args.data_path + '/sequences.pt')[:,:1]
    except Exception as e:
        raise e
        sequences = None
        input_size = inputs.shape[1]
    num_batches = (inputs.size(0) // args.batch_size) + 1

    with torch.no_grad():
        for j in range(2):
            predictions = []
            for i in range(num_batches):
                batch_inputs = inputs[args.batch_size*i:args.batch_size*(i+1)]
                indices = np.nonzero(batch_inputs[:,0,0].numpy() == seq_labels.numpy())[0]
                batch_inputs = torch.cat([batch_inputs[:,1:,:], sequences[indices]], dim=1).cuda()
                predictions += [model(batch_inputs, use_classifier=j).cpu().squeeze()]
            predictions = torch.cat(predictions).numpy()
            model_name = args.model_path[args.model_path.find('/'):args.model_path.rfind('.')]
            submission_title = 'submissions/' + args.model_path[args.model_path.find('/')+1:args.model_path.rfind('/')]
            submit(predictions, submission_title + '_%d' % j)

            mean = predictions.mean()
            var = predictions.std()

            print('Submission mean: %.3f' % mean)
            print('Submission standard deviation: %.3f' % var)
