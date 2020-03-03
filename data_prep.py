import numpy as np
import torch

# Keys to npzfile of train & eval
train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058',
'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097',
'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']


## Training Data ##

# Load data
print('Loading Training Data...')
train_data = np.load('train.npz')
print('Loading complete.')

# Combine Train Data to use information from all cells
train_inputs = [] # Input histone mark data
train_outputs = [] # Correct expression value
test_inputs = [] # Input histone mark data
test_outputs = [] # Correct expression value
for cell in train_cells[:-2]:
    cell_data = train_data[cell]
    hm_data = cell_data[:-1000,:,1:6]
    exp_values = cell_data[:-1000,0,6]
    train_inputs.append(hm_data)
    train_outputs.append(exp_values)
    hm_data = cell_data[-1000:,:,1:6]
    exp_values = cell_data[-1000:,0,6]
    test_inputs.append(hm_data)
    test_outputs.append(exp_values)
for cell in train_cells[-2:]:
    cell_data = train_data[cell]
    hm_data = cell_data[:,:,1:6]
    exp_values = cell_data[:,0,6]
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
torch.save(torch.from_numpy(train_inputs).permute(0, 2, 1), 'train_in.pt')
torch.save(torch.from_numpy(train_outputs), 'train_out.pt')
torch.save(torch.from_numpy(test_inputs).permute(0, 2, 1), 'test_in.pt')
torch.save(torch.from_numpy(test_outputs), 'test_out.pt')
print('Saving Complete.')


## Eval Data ##

# Load data
print('Loading Submission Data...')
eval_data = np.load('eval.npz')

# Prepare Eval inputs in similar way
eval_inputs = []
for cell in eval_cells:
    cell_data = eval_data[cell]
    hm_data = cell_data[:,:,1:6]
    eval_inputs.append(hm_data)

eval_inputs = np.concatenate(eval_inputs, axis=0)
print('Submission inputs shape: %s' % str(eval_inputs.shape))

# Save data
print('Saving Submission Data...')
torch.save(torch.from_numpy(eval_inputs).permute(0, 2, 1), 'submission_in.pt')
print('Saving Complete.')
