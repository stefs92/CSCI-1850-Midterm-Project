import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self, norm_mean=0, norm_std=1):
        super(Model, self).__init__()
        self.register_buffer('norm_mean', norm_mean)
        self.register_buffer('norm_std', norm_std)

    def norm(self, values):
        return (values - self.norm_mean) / self.norm_std


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


class ConvolutionalModel(Model):
    def __init__(self, norm_mean=0, norm_std=1, n_outs=1):
        super(ConvolutionalModel, self).__init__(norm_mean, norm_std)
        self.conv1   = ConvBlock(5, 64, 3, batch_norm=True, dropout=.1, stride=2)
        self.conv2   = ConvBlock(64, 64, 3, batch_norm=True, dropout=.1, stride=1)
        self.conv3   = ConvBlock(64, 128, 3, batch_norm=True, dropout=.2, stride=2)
        self.conv4   = ConvBlock(128, 128, 3, batch_norm=True, dropout=.2, stride=1)
        self.conv5   = ConvBlock(128, 256, 3, batch_norm=True, dropout=.3, stride=2)
        self.conv6   = ConvBlock(256, 256, 3, batch_norm=True, dropout=.3, stride=1)
        self.conv7   = ConvBlock(256, 512, 3, batch_norm=True, dropout=.4, stride=2)
        self.conv8   = ConvBlock(512, 512, 3, batch_norm=True, dropout=.4, stride=1)
        self.conv9   = ConvBlock(512, n_outs, 1)

    def forward(self, inputs, show=False, eval=False):
        x0 = self.norm(inputs)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8).view(inputs.size(0), -1)
        if show:
            return x0, x1, x2, x3, x4, x5, x6, x7, x8, x9
        else:
            return x9


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.classifier = None
        if isinstance(models[-1], ClassificationModel):
            self.models = nn.ModuleList(models[:-1])
            self.classifier = models[-1]
        else:
            self.models = nn.ModuleList(models)

    def forward(self, inputs):
        predictions = [model(inputs, eval=True) for model in self.models]
        shape = predictions[0].shape
        predictions = torch.cat([pred.view(shape[0], -1, 1) for pred in predictions], dim=2)
        if self.classifier is not None:
            weights = F.softmax(self.classifier(inputs), dim=1).unsqueeze(1)
            mean_prediction = torch.sum(predictions * weights, dim=2).view(shape)
        else:
            mean_prediction = torch.mean(predictions, dim=2).view(shape)
        return mean_prediction


class ClassificationModel(Model):
    def __init__(self, norm_mean=0, norm_std=1, n_classes=48):
        super(ClassificationModel, self).__init__(norm_mean, norm_std)
        self.conv_stack = ConvolutionalModel(norm_mean, norm_std, n_classes)

    def forward(self, inputs, show=False, eval=False):
        activations = self.conv_stack(inputs, show)
        if show:
            activations[-1] = activations[-1]
        else:
            activations = activations
        return activations


class FactoredModel(Model):
    def __init__(self, norm_mean=0, norm_std=1):
        super(FactoredModel, self).__init__(norm_mean, norm_std)
        self.conv_stack = ConvolutionalModel(norm_mean, norm_std, 2)

    def forward(self, inputs, show=False, eval=False):
        activations = self.conv_stack(inputs, show)
        act = activations
        if show:
            act = activations[-1]
        act = torch.stack([torch.relu(act[:, 0]), F.sigmoid(act[:, 1])], dim=1)
        if eval:
            act = act[:, 0] * (act[:, 1] * 2 - 1)
        if show:
            act = list(activations[:-1]) + [act]
        return act
