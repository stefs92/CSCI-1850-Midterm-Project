import numpy as np
import torch
import argparse
import os
from collections import defaultdict

# Keys to npzfile of train & eval
train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058',
'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097',
'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

seq_dict = {'N':4, 'A':0, 'G':1, 'T':2, 'C':3}

def extract_data(cell_data, sequence_labels, sequence_data):
    # HM data
    hm_data = cell_data[:,:,1:6]
    mean_data = [np.expand_dims(hm_data.mean(axis=0), 0)]*hm_data.shape[0]
    mean_data = np.concatenate(mean_data, axis=0)
    std_data = [np.expand_dims(hm_data.std(axis=0), 0)]*hm_data.shape[0]
    std_data = np.concatenate(std_data, axis=0)
    input_hm = np.concatenate([cell_data[:,:,:1], hm_data, mean_data, std_data], axis=2)

    # Expression data
    try:
        output_data = cell_data[:,0,6]
    except:
        output_data = None
    return input_hm, output_data


def build_motifs(length):
    motifs = [[], ['']]
    for l in range(length):
        for motif in motifs[1]:
            for c in 'NAGTC':
                motifs[0] += [motif+c]
        motifs = [[], motifs[0]]
    motif_dict = defaultdict(int)
    for i, motif in enumerate(motifs[1]):
        motif_dict[motif] = i
    return motif_dict


def one_hot(tensor, length):
    tensors = []
    for i in range(5**length):
        #if i in tensor.unique():
        tensors += [(tensor == i).float()]
        #else:
        #    tensors += [torch.zeros(tensor.shape)]
    return torch.stack(tensors, dim=-1)


def extract_motifs(sequences, motif_length):
    motif_sequences = []
    motifs = build_motifs(motif_length)
    for sequence in sequences:
        motif_sequence = []
        for i in range(len(sequence) - motif_length + 1):
            motif_sequence += [motifs[sequence[i:i+motif_length]]]
        motif_sequences += [motif_sequence]
    return one_hot(torch.LongTensor(motif_sequences), motif_length)


def count_motifs(sequences, motif_length, window_size, stride):
    n_windows = (sequences.shape[1]-window_size)//stride + 1
    counts = torch.zeros(len(sequences), n_windows, sequences.shape[2])
    for i in range(sequences.shape[0]):
        for j in range(n_windows):
            counts[i,j] = torch.sum(sequences[i,j*stride:j*stride+window_size], dim=0)
    return counts


def process_sequences(sequences, motif_length, window_size, stride):
    sequences = extract_motifs(sequences, motif_length)
    if window_size > 1:
        motif_counts = count_motifs(sequences, motif_length, window_size, stride)
    else:
        motif_counts = sequences[:,::stride]
    return torch.log(motif_counts + 1)


def save_tensors(hm, out, name, augment=True):
    hm = torch.from_numpy(hm).permute(0, 2, 1)
    if args.threshold > 0:
        hm[:,1:][hm[:,1:] < args.threshold] = 0
    if args.noise > 0 and augment:
        hm[:,1:] = hm[:,1:] + torch.randn_like(hm[:,1:]) * args.noise
    with torch.no_grad():
        for i in range(args.smooth):
            hm[:,1:] = smooth(hm[:,1:])

    torch.save(hm, args.destination + '/%s_in.pt' % name)
    if out is not None:
        torch.save(torch.from_numpy(out), args.destination + '/%s_out.pt' % name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the training and testing datasets.')
    parser.add_argument('destination', type=str, help='Path to save new dataset.')
    parser.add_argument('-n_test', type=int, default=2, help='Number of cells to reserve for the test set.')
    parser.add_argument('-threshold', type=int, default=0, help='Minimum threshold for input values.')
    parser.add_argument('-smooth', type=int, default=0, help='Number of smoothing passes over the inputs.')
    parser.add_argument('-noise', type=float, default=0, help='Amount of added normal noise (train data only).')
    parser.add_argument('-batch_size', type=int, default=250, help='Batch size for sequence processing.')
    parser.add_argument('-sequence_threshold', type=int, default=10000, help='Threshold for dataset-wide motif counts.')
    parser.add_argument('-motif_length', type=int, default=1, help='Length of motifs to extract.')
    parser.add_argument('-window_size', type=int, default=1, help='Size of windows to count motifs over.')
    parser.add_argument('-stride', type=int, default=1, help='Stride of sliding-window motif counting.')

    args = parser.parse_args()

    print('Preprocessing Sequence Data...')
    padding = args.motif_length-1
    padding = (padding // 2, padding // 2 + (padding % 2))
    padding = ('N'*padding[0], 'N'*padding[1])
    seq_data = np.genfromtxt('seq_data.csv', delimiter=',', dtype=None)
    seq_labels = torch.LongTensor([x[0] for x in seq_data]).view(-1, 1)
    text_sequences = [padding[0]+x[1].decode('utf-8')+padding[1] for x in seq_data]
    if args.motif_length == 1 and args.window_size == 1 and args.stride == 1:
        sequences = torch.LongTensor([[seq_dict[c] for c in x] for x in text_sequences])
    else:
        sequences = []
        n_batches = len(text_sequences)//args.batch_size + 1
        for i in range(n_batches):
            print('processing batch %d of %d' % (i+1, n_batches), end='\r')
            sequences += [process_sequences(text_sequences[args.batch_size*i:args.batch_size*(i+1)], args.motif_length, args.window_size, args.stride)]
        sequences = torch.cat(sequences, dim=0)
        sequences = sequences[:, :, sequences.sum(dim=(0,1)) > 10000]
    print('')

    # Translate from integer counts to normalized scale used by dataset
    args.threshold = torch.log(torch.FloatTensor([args.threshold + 1]))

    # Prep smoothing "function"
    k = 3
    smooth = torch.nn.Conv1d(15, 15, k, padding=((k-1)//2))
    smooth.bias[:] = 0
    for i in range(5):
        smooth.weight[i] = torch.zeros_like(smooth.weight[i])
        smooth.weight[i][i] = 1 / k

    ## Training Data ##

    # Create destination, only if it does not exist
    os.makedirs(args.destination)
    print('Saving Sequence Data...')
    torch.save(sequences, args.destination + '/sequences.pt')
    torch.save(seq_labels, args.destination + '/sequence_labels.pt')

    # shuffle cells
    i_shuffle = torch.randperm(len(train_cells))
    train_cells = [train_cells[i] for i in i_shuffle]

    # Load data
    print('Loading Training Data...')
    train_data = np.load('train.npz')

    # Combine Train Data to use information from all cells
    train_seq = [] # Input sequence data
    train_hm = [] # Input histone mark data
    train_outputs = [] # Correct expression value
    test_seq = [] # Input sequence data
    test_hm = [] # Input histone mark data
    test_outputs = [] # Correct expression value
    split_index = -1 * args.n_test
    for cell in train_cells[:split_index]:
        cell_data = train_data[cell]
        hm_data, exp_values = extract_data(cell_data[:-1000], seq_labels, sequences)
        train_hm.append(hm_data)
        train_outputs.append(exp_values)
        hm_data, exp_values = extract_data(cell_data[-1000:], seq_labels, sequences)
        test_hm.append(hm_data)
        test_outputs.append(exp_values)
    for cell in train_cells[split_index:]:
        cell_data = train_data[cell]
        hm_data, exp_values = extract_data(cell_data, seq_labels, sequences)
        test_hm.append(hm_data)
        test_outputs.append(exp_values)

    train_hm = np.concatenate(train_hm, axis=0)
    train_outputs = np.concatenate(train_outputs, axis=0)
    print('Training histones shape: %s' % str(train_hm.shape))
    print('Training outputs shape: %s' % str(train_outputs.shape))
    test_hm = np.concatenate(test_hm, axis=0)
    test_outputs = np.concatenate(test_outputs, axis=0)
    print('Test histones shape: %s' % str(test_hm.shape))
    print('Test outputs shape: %s' % str(test_outputs.shape))

    # Save data
    print('Saving Training and Test Data...')
    save_tensors(train_hm, train_outputs, 'train')
    save_tensors(test_hm, test_outputs, 'test', augment=False)
    print('Saving Complete.')


    ## Submission Data ##

    # Load data
    print('Loading Submission Data...')
    eval_data = np.load('eval.npz')

    # Prepare Eval inputs in similar way
    eval_seq = []
    eval_hm = []
    for cell in eval_cells:
        cell_data = eval_data[cell]
        hm_data, _ = extract_data(cell_data, seq_labels, sequences)
        eval_hm.append(hm_data)

    eval_hm = np.concatenate(eval_hm, axis=0)
    print('Submission histones shape: %s' % str(eval_hm.shape))

    # Save data
    print('Saving Submission Data...')
    save_tensors(eval_hm, None, 'submission', augment=False)
    print('Saving Complete.')
