import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, *args, batch_norm=False, dropout=0, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(args[1])
        else:
            self.batch_norm = False

    def forward(self, inputs):
        if self.batch_norm:
            return F.leaky_relu(self.dropout(self.batch_norm(self.conv(inputs))))
        else:
            return F.leaky_relu(self.dropout(self.conv(inputs)))



class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.conv1   = ConvBlock(5, 64, 3, batch_norm=True, dropout=.2, stride=2)
        self.conv2   = ConvBlock(64, 64, 3, batch_norm=True, dropout=.2, stride=1)
        self.conv3   = ConvBlock(64, 128, 3, batch_norm=True, dropout=.2, stride=2)
        self.conv4   = ConvBlock(128, 128, 3, batch_norm=True, dropout=.2, stride=1)
        self.layer = nn.TransformerEncoderLayer(128, 8, dropout=.2)
        self.encoder = nn.TransformerEncoder(self.layer, 1)
        self.linear1 = nn.Linear(2688, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(.5)

    def forward(self, inputs):
        '''
        # Convolution
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Transformer
        x = x.permute(2, 0, 1)
        x = F.leaky_relu(self.encoder(x)).permute(1, 2, 0)
        # MLP
        x = x.reshape(x.size(0), -1)
        x = F.leaky_relu(self.dropout(self.linear1(x)))
        x = F.leaky_relu(self.dropout(self.linear2(x)))
        x = self.linear3(x)
        '''
        x = torch.ones(inputs.size(0)).to(inputs) * -0.2724
        return x

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(5, 512)
        self.layer = nn.TransformerEncoderLayer(512, 8)
        self.encoder = nn.TransformerEncoder(self.layer, 2)
        self.decoder = nn.Linear(512, 1)

    def forward(self, inputs):
        x = F.leaky_relu(self.embedding(inputs.permute(0, 2, 1)))
        x = F.leaky_relu(self.encoder(x.permute(1, 0, 2))).permute(1, 2, 0)
        x = self.decoder(x[:, :, 0])
        return x


class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()
        self.conv1   = ConvBlock(5, 64, 3, batch_norm=True, dropout=.2, stride=2)
        self.conv2   = ConvBlock(64, 64, 3, batch_norm=True, dropout=.2, stride=1)
        self.conv3   = ConvBlock(64, 128, 3, batch_norm=True, dropout=.2, stride=2)
        self.conv4   = ConvBlock(128, 128, 3, batch_norm=True, dropout=.2, stride=1)
        self.conv5   = ConvBlock(128, 256, 3, batch_norm=True, dropout=.2, stride=2)
        self.conv6   = ConvBlock(256, 256, 3, batch_norm=True, dropout=.2, stride=1)
        self.conv7   = ConvBlock(256, 512, 3, batch_norm=True, dropout=.2, stride=2)
        self.conv8   = ConvBlock(512, 512, 3, batch_norm=True, dropout=.2, stride=1)
        self.linear1 = nn.Linear(2688, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(.5)

    def forward(self, inputs):
        # Convolution
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        # MLP
        #x = F.leaky_relu(self.dropout(self.linear1(x)))
        #x = F.leaky_relu(self.dropout(self.linear2(x)))
        x = self.linear3(x)
        return x


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



def build_model(args):
    return ConvolutionalModel().cuda()


def shuffle_data(inputs, outputs):
    i_shuffle = torch.randperm(inputs.size(0))
    return inputs[i_shuffle], outputs[i_shuffle]


def step(model, inputs, outputs, loss_f, opt):
    predictions = model(inputs).view(-1)
    loss = loss_f(predictions, outputs)
    torch.mean(loss).backward()
    opt.step()
    opt.zero_grad()
    return loss


def train(model, inputs, outputs, args):
    bs = args.batch_size
    inputs, outputs = shuffle_data(inputs, outputs)
    data_size = inputs.size(0)
    partition_size = data_size // args.partitions
    loss_f = nn.MSELoss(reduction='none')
    eval_f = nn.MSELoss(reduction='sum')

    ''' K-fold Cross Validation'''
    hold_model = model
    fold_final_eval_losses = torch.zeros(args.partitions)
    fold_final_train_losses = torch.zeros(args.partitions)
    fold_models = [copy.deepcopy(hold_model) for i in range(args.partitions)]
    fold_opts = [optim.Adam(model.parameters(), lr=args.learning_rate) for model in fold_models]
    mean_loss = 0
    for i_fold in range(args.partitions):
        args.batch_size = bs
        model = fold_models[i_fold]
        opt = fold_opts[i_fold]
        eval_inputs = inputs[i_fold*partition_size:(i_fold+1)*partition_size]
        eval_outputs = outputs[i_fold*partition_size:(i_fold+1)*partition_size]
        train_inputs = torch.cat([inputs[:i_fold*partition_size], inputs[(i_fold+1)*partition_size:]])
        train_outputs = torch.cat([outputs[:i_fold*partition_size], outputs[(i_fold+1)*partition_size:]])
        data_mean = torch.mean(eval_outputs).detach()
        data_error = eval_f(eval_outputs.detach(), torch.ones(eval_outputs.size(0))*data_mean).detach() / eval_outputs.size(0)

        n_batches = (train_inputs.size(0) // args.batch_size)+1
        train_losses = torch.zeros(n_batches)
        if args.loss_sampling:
            logits = torch.zeros(train_inputs.size(0))
        for i_epoch in range(args.epochs):
            args.batch_size = int(args.batch_size * 1.01)
            model.train()
            train_inputs, train_outputs = shuffle_data(train_inputs, train_outputs)
            for i_batch in range(n_batches):
                if args.loss_sampling and i_epoch > 0:
                    batch_indices = torch.multinomial(logits, args.batch_size, replacement=True)
                else:
                    batch_indices = slice(i_batch*args.batch_size, (i_batch+1)*args.batch_size)

                batch_inputs = train_inputs[batch_indices].cuda()
                batch_outputs = train_outputs[batch_indices].cuda()
                if batch_inputs.size(0) == 0:
                    continue
                batch_losses = step(model, batch_inputs, batch_outputs, loss_f, opt)
                train_losses[i_batch] = torch.mean(batch_losses).item()

                if args.loss_sampling:
                    with torch.no_grad():
                        logits[batch_indices] = batch_losses.cpu()

            '''

            Thought:
            If we take all of the eval predictions and labels together, we could
            do some sort of max over potential cutoffs of categorization error
            to get a lower bound on the accuracy over the eval set.

            '''

            with torch.no_grad():
                model.eval()
                eval_inputs, eval_outputs = shuffle_data(eval_inputs, eval_outputs)
                n_batches = (eval_inputs.size(0) // args.batch_size)+1
                sum_loss = 0

                for i_batch in range(n_batches):
                    batch_inputs = eval_inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
                    batch_outputs = eval_outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
                    if batch_inputs.size(0) == 0:
                        continue
                    predictions = model(batch_inputs).view(-1)
                    sum_loss += eval_f(predictions, batch_outputs).item()

                mean_loss = sum_loss / eval_inputs.size(0)
                print('Fold %d, Epoch %d Mean Train / Eval Loss and R^2 Value: %.3f / %.3f / %.3f ' % (i_fold+1, i_epoch+1, torch.mean(train_losses), mean_loss, 1 - mean_loss / data_error), end='\r')
        fold_final_eval_losses[i_fold] = mean_loss
        fold_final_train_losses[i_fold] = torch.mean(train_losses).detach()
        fold_models[i_fold] = model
        print('') # to keep only the final epoch losses from each fold


    final_mean_eval_loss = torch.mean(fold_final_eval_losses)
    final_mean_train_loss = torch.mean(fold_final_train_losses)
    print(('Mean Train / Eval Loss Across Folds at %d Epochs: %.3f / %.3f' % (args.epochs, final_mean_train_loss, final_mean_eval_loss))+' '*10)

    ''' Ensembling '''
    with torch.no_grad():
        inputs, outputs = shuffle_data(inputs, outputs)
        n_batches = (inputs.size(0) // args.batch_size)+1
        sum_loss = [0]*(args.partitions+1)

        for i_batch in range(n_batches):
            batch_inputs = inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
            batch_outputs = outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
            if batch_inputs.size(0) == 0:
                continue
            predictions = torch.cat([model(batch_inputs).view(-1, 1) for model in fold_models], dim=1)
            sum_loss[:-1] = [sum_loss[i] + eval_f(predictions[:,i], batch_outputs).item() for i in range(args.partitions)]
            predictions = torch.mean(predictions, dim=1).view(-1)
            sum_loss[-1] += eval_f(predictions, batch_outputs).item()

        mean_loss = [sum_loss[i] / inputs.size(0) for i in range(args.partitions+1)]
        print(('Eval Loss Of Ensemble Across Folds at %d Epochs: %s' % (args.epochs, str(mean_loss)))+' '*10)



    ''' Training on All Data '''
    [model.train() for model in fold_models]

    n_batches = (inputs.size(0) // args.batch_size)+1
    if args.loss_sampling:
        logits = torch.zeros(inputs.size(0))
    i_epoch = -1
    try:
        while True:
            i_epoch += 1
            train_losses = torch.zeros(len(fold_models), n_batches)

            train_inputs, train_outputs = shuffle_data(inputs, outputs)
            for i_batch in range(n_batches):
                if args.loss_sampling and i_epoch > 0:
                    batch_indices = torch.multinomial(logits, args.batch_size, replacement=True)
                else:
                    batch_indices = slice(i_batch*args.batch_size, (i_batch+1)*args.batch_size)

                batch_inputs = train_inputs[batch_indices].cuda()
                batch_outputs = train_outputs[batch_indices].cuda()
                for i_model in range(len(fold_models)):
                    batch_losses = step(fold_models[i_model], batch_inputs, batch_outputs, loss_f, fold_opts[i_model])
                    train_losses[i_model, i_batch] = torch.mean(batch_losses).item()

                if args.loss_sampling:
                    with torch.no_grad():
                        logits[batch_indices] = batch_losses.cpu()

            print('All Data Epoch %d Mean Train Loss: %s' % (i_epoch+1, str(torch.mean(train_losses, dim=1).numpy())), end='\r')
    except Exception as e:
        pass
    finally:
        return EnsembleModel(fold_models)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a prediction over the eval set.')
    parser.add_argument('model_path', type=str, help='Relative path at which to save the model.')
    parser.add_argument('-partitions', type=int, default=3, help='Number of partitions for cross-fold validation.')
    parser.add_argument('-batch_size', type=int, default=256, help='Number of samples per batch.')
    parser.add_argument('-batch_size_annealing', type=float, default=1.0, help='Per-batch multiplier on batch size.')
    parser.add_argument('-epochs', type=int, default=25, help='Number of epochs to train for.')
    parser.add_argument('-learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('-loss_sampling', default=False, action='store_true', dest='loss_sampling', help='Flag to use loss-based sampling in place of traditional batching.')

    args = parser.parse_args()

    try:
        model = torch.load(args.model_path).cuda()
    except:
        model = build_model(args)
    inputs = torch.load('train_in.pt')
    outputs = torch.load('train_out.pt')

    model = train(model, inputs, outputs, args)

    torch.save(model.cpu(), args.model_path)
