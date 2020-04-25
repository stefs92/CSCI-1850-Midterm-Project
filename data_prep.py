import numpy as np
import torch
import argparse
import os

# Keys to npzfile of train & eval
train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058',
'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097',
'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

def extract_data(cell_data):
    hm_data = cell_data[:,:,1:6]
    mean_data = [np.expand_dims(hm_data.mean(axis=0), 0)]*hm_data.shape[0]
    mean_data = np.concatenate(mean_data, axis=0)
    std_data = [np.expand_dims(hm_data.std(axis=0), 0)]*hm_data.shape[0]
    std_data = np.concatenate(std_data, axis=0)
    input_data = np.concatenate([hm_data, mean_data, std_data], axis=2)
    try:
        output_data = cell_data[:,0,6]
    except:
        output_data = None
    return input_data, output_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the training and testing datasets.')
    parser.add_argument('destination', type=str, help='Path to save new dataset.')
    parser.add_argument('-n_test', type=int, default=2, help='Number of cells to reserve for the test set.')
    parser.add_argument('-threshold', type=int, default=0, help='Minimum threshold for input values.')
    parser.add_argument('-smooth', type=int, default=0, help='Number of smoothing passes over the inputs.')
    parser.add_argument('-noise', type=float, default=0, help='Amount of added normal noise (train data only).')

    args = parser.parse_args()

    '''
    np.genfromtxt('final_project_data/seq_data.csv', delimiter=',', dtype=None)
    '''

    # Translate from integer counts to normalized scale used by dataset
    args.threshold = torch.log(torch.FloatTensor([args.threshold + 1]))

    # Prep smoothing "function"
    k = 3
    smooth = torch.nn.Conv1d(5, 5, k, padding=((k-1)//2))
    smooth.bias[:] = 0
    for i in range(5):
        smooth.weight[i] = torch.zeros_like(smooth.weight[i])
        smooth.weight[i][i] = 1 / k

    ## Training Data ##

    # Create destination, only if it does not exist
    os.makedirs(args.destination)

    # shuffle cells
    i_shuffle = torch.randperm(len(train_cells))
    train_cells = [train_cells[i] for i in i_shuffle]

    # Load data
    print('Loading Training Data...')
    train_data = np.load('train.npz')

    # Combine Train Data to use information from all cells
    train_inputs = [] # Input histone mark data
    train_outputs = [] # Correct expression value
    test_inputs = [] # Input histone mark data
    test_outputs = [] # Correct expression value
    split_index = -1 * args.n_test
    for cell in train_cells[:split_index]:
        cell_data = train_data[cell]
        hm_data, exp_values = extract_data(cell_data[:-1000])
        train_inputs.append(hm_data)
        train_outputs.append(exp_values)
        hm_data, exp_values = extract_data(cell_data[-1000:])
        test_inputs.append(hm_data)
        test_outputs.append(exp_values)
    for cell in train_cells[split_index:]:
        cell_data = train_data[cell]
        hm_data, exp_values = extract_data(cell_data)
        test_inputs.append(hm_data)
        test_outputs.append(exp_values)

    train_inputs = np.concatenate(train_inputs, axis=0)
    train_outputs = np.concatenate(train_outputs, axis=0)
    print('Training inputs shape: %s' % str(train_inputs.shape))
    print('Training outputs shape: %s' % str(train_outputs.shape))
    test_inputs = np.concatenate(test_inputs, axis=0)
    test_outputs = np.concatenate(test_outputs, axis=0)
    print('Test inputs shape: %s' % str(test_inputs.shape))
    print('Test outputs shape: %s' % str(test_outputs.shape))

    # Save data
    print('Saving Training and Test Data...')
    train_inputs = torch.from_numpy(train_inputs).permute(0, 2, 1)
    train_inputs[train_inputs < args.threshold] = 0
    train_inputs = train_inputs + torch.randn_like(train_inputs) * args.noise
    with torch.no_grad():
        for i in range(args.smooth):
            train_inputs = smooth(train_inputs)

    test_inputs = torch.from_numpy(test_inputs).permute(0, 2, 1)
    test_inputs[test_inputs < args.threshold] = 0
    test_inputs = test_inputs + torch.randn_like(test_inputs) * args.noise
    with torch.no_grad():
        for i in range(args.smooth):
            test_inputs = smooth(test_inputs)

    torch.save(train_inputs, args.destination + '/train_in.pt')
    torch.save(torch.from_numpy(train_outputs), args.destination + '/train_out.pt')
    torch.save(test_inputs, args.destination + '/test_in.pt')
    torch.save(torch.from_numpy(test_outputs), args.destination + '/test_out.pt')
    print('Saving Complete.')


    ## Submission Data ##

    # Load data
    print('Loading Submission Data...')
    eval_data = np.load('eval.npz')

    # Prepare Eval inputs in similar way
    eval_inputs = []
    for cell in eval_cells:
        cell_data = eval_data[cell]
        hm_data, _ = extract_data(cell_data)
        eval_inputs.append(hm_data)

    eval_inputs = np.concatenate(eval_inputs, axis=0)
    print('Submission inputs shape: %s' % str(eval_inputs.shape))

    # Save data
    print('Saving Submission Data...')
    eval_inputs = torch.from_numpy(eval_inputs).permute(0, 2, 1)
    eval_inputs[eval_inputs < args.threshold] = 0
    with torch.no_grad():
        for i in range(args.smooth):
            eval_inputs = smooth(eval_inputs)

    torch.save(eval_inputs, args.destination + '/submission_in.pt')
    print('Saving Complete.')
