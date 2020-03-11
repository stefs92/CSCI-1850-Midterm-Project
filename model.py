import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, *args, batch_norm=False, dropout=0, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
        self.dropout = nn.Dropout2d(dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(args[1])
        else:
            self.batch_norm = False

    def forward(self, inputs):
        if self.batch_norm:
            return F.leaky_relu(self.dropout(self.batch_norm(self.conv(inputs))))
        else:
            return F.leaky_relu(self.dropout(self.conv(inputs)))


class ConvolutionalModel(nn.Module):
    def __init__(self, norm_mean=0, norm_std=1):
        super(ConvolutionalModel, self).__init__()
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.conv1   = ConvBlock(5, 64, 3, batch_norm=True, dropout=.1, stride=2)
        self.conv2   = ConvBlock(64, 64, 3, batch_norm=True, dropout=.1, stride=1)
        self.conv3   = ConvBlock(64, 128, 3, batch_norm=True, dropout=.2, stride=2)
        self.conv4   = ConvBlock(128, 128, 3, batch_norm=True, dropout=.2, stride=1)
        self.conv5   = ConvBlock(128, 256, 3, batch_norm=True, dropout=.3, stride=2)
        self.conv6   = ConvBlock(256, 256, 3, batch_norm=True, dropout=.3, stride=1)
        self.conv7   = ConvBlock(256, 512, 3, batch_norm=True, dropout=.4, stride=2)
        self.conv8   = ConvBlock(512, 512, 3, batch_norm=True, dropout=.4, stride=1)
        self.conv9   = ConvBlock(512, 1, 1)

    def forward(self, inputs, show=False):
        x0 = (inputs - self.norm_mean) / self.norm_std
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        if show:
            return x0, x1, x2, x3, x4, x5, x6, x7, x8, x9
        else:
            return x9.view(x9.size(0), -1)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, inputs):
        predictions = [model(inputs) for model in self.models]
        shape = predictions[0].shape
        predictions = torch.cat([pred.view(shape[0], -1, 1) for pred in predictions], dim=2)
        mean_prediction = torch.mean(predictions, dim=2).view(shape)
        return mean_prediction
