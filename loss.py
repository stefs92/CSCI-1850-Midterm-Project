import torch
import torch.nn as nn
import torch.nn.functional as F


class FactoredLoss(nn.Module):
    def __init__(self, reduction=True):
        super(FactoredLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.ce = nn.BCELoss(reduction='none')
        self.reduction = reduction

    def forward(self, inputs, outputs):
        mag_loss = self.mse(inputs[:, 0], torch.abs(outputs))
        sign_loss = self.ce(inputs[:, 1], (outputs > 0).float())
        full_loss = mag_loss + sign_loss * torch.abs(outputs)
        if self.reduction:
            full_loss = torch.sum(full_loss)
        return full_loss
