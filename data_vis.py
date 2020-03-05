import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


train_inputs = torch.load('train_in.pt').contiguous().view(-1, 1, 500)
train_outputs = torch.load('train_out.pt').view(-1)
test_inputs = torch.load('test_in.pt').contiguous().view(-1, 1, 500)
test_outputs = torch.load('test_out.pt').view(-1)

smoothing_factor = 8
conv = nn.Conv1d(1, 1, smoothing_factor, bias=False)
conv.weight = nn.Parameter(torch.ones(1, 1, smoothing_factor) / smoothing_factor)

with torch.no_grad():
    train_inputs = conv(train_inputs).squeeze()
    train_inputs = train_inputs - torch.mean(train_inputs)
    train_outputs = train_outputs - torch.mean(train_outputs)
    test_inputs = conv(test_inputs).squeeze()
    test_inputs = test_inputs - torch.mean(train_inputs)
    test_outputs = test_outputs - torch.mean(train_outputs)

train_inputs = train_inputs.numpy()
train_outputs = train_outputs.numpy()
test_inputs = test_inputs.numpy()
test_outputs = test_outputs.numpy()

pca = PCA(25)
pca.fit(train_inputs)
pca_train_inputs = pca.transform(train_inputs)
pca_test_inputs = pca.transform(test_inputs)

plt.imshow(np.concatenate([pca_train_inputs[::4000], train_outputs[::4000].reshape(-1, 1)*10], axis=1))
plt.show()
plt.imshow(np.concatenate([pca_train_inputs[::8000, :10], train_outputs[::8000].reshape(-1, 1)*10], axis=1))
plt.show()
plt.imshow(np.concatenate([pca_train_inputs[::16000, :3], train_outputs[::16000].reshape(-1, 1)*10], axis=1))
plt.show()

pos_inds = train_outputs > 3
neg_inds = train_outputs < -3
pos_examples = np.concatenate([train_inputs[pos_inds][:200], train_outputs[pos_inds][:200].reshape(-1, 1)], axis=1)
neg_examples = np.concatenate([train_inputs[neg_inds][:200], train_outputs[neg_inds][:200].reshape(-1, 1)], axis=1)
plt.imshow(np.concatenate([pos_examples, neg_examples], axis=0))
plt.show()

pos_inds = test_outputs > 3
neg_inds = test_outputs < -3
pos_examples = np.concatenate([test_inputs[pos_inds][:200], test_outputs[pos_inds][:200].reshape(-1, 1)], axis=1)
neg_examples = np.concatenate([test_inputs[neg_inds][:200], test_outputs[neg_inds][:200].reshape(-1, 1)], axis=1)
plt.imshow(np.concatenate([pos_examples, neg_examples], axis=0))
plt.show()
