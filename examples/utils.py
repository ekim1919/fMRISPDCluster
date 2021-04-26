import argparse
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from dataset import *
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
from spdnet.optimizer import StiefelMetaOptimizer
from argparse import Namespace

class recurrent_layer(nn.Module):
    def __init__(self):
        super(recurrent_layer, self).__init__()
        self.rect1  = SPDRectified()
        #self.rect2  = SPDRectified()
        #self.rect3  = SPDRectified()

    def forward(self, x):
        x = self.rect1(x)
        #x = self.rect2(x)
        #x = self.rect3(x)
        return x

class input_layer(nn.Module):
    def __init__(self, hs):
        super(input_layer, self).__init__()
        self.trans1 = SPDTransform(57, 20)
        self.trans2 = SPDTransform(20, 10)
        self.trans3 = SPDTransform(10, hs)
        self.rect1  = SPDRectified()
        self.rect2  = SPDRectified()
        self.rect3  = SPDRectified()

    def forward(self, x):
        x = self.trans1(x)
        x = self.rect1(x)
        x = self.trans2(x)
        x = self.rect2(x)
        x = self.trans3(x)
        x = self.rect3(x)
        return x

class output_layer(nn.Module):
    def __init__(self, hs):
        super(output_layer, self).__init__()
        self.tangent = SPDTangentSpace(hs)
        self.linear = nn.Linear(int(hs*(hs+1)/2), 17, bias=True)

    def forward(self, x):
        x = self.tangent(x)
        x = self.linear(x)
        return x

class RSPDNet(nn.Module):
    def __init__(self, hs, use_cuda):
        super(RSPDNet, self).__init__()
        self.input_layer = input_layer(hs)
        self.output_layer = output_layer(hs)
        self.recurrent_layer = recurrent_layer()
        self.alpha = torch.nn.Parameter(torch.randn(1))
        self.alpha.requires_grad = True
        self.use_cuda = use_cuda
        self.act = nn.Sigmoid()

    def forward(self, input, future=0):
        outputs = []
        # The two corresponds to the dimension of the recurrent SPD matrix.
        h_t = torch.zeros(input.size(0), 2, 2, dtype=torch.float)
        if self.use_cuda:
            h_t = h_t.cuda()

        for i in range(input.size(1)):
            x = input[:, i, :, :]
            tmp_a = self.act(self.alpha)*self.input_layer(x)
            tmp_b = (1-self.act(self.alpha))*self.recurrent_layer(h_t)
            h_t = tmp_a + tmp_b
            output = self.output_layer(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1)
        return outputs

def sliding_window(x, ws=30):
    d, n = x.shape
    return np.stack([(1/n)*np.matmul(x[:, i:ws+i], x[:, i:ws+i].T) for i in range(n-ws)])

def get_sliding_labels(x, ws=30):
    n = x.shape[0]
    return np.stack([x[int(ws/2)+i] for i in range(n-ws)])

class SPDRNNDataset(Dataset):
    def __init__(self, shuffle=False, scan_id=1, ws=30):
        super(SPDRNNDataset, self).__init__()
        self.scan_id = scan_id
        self.ws = ws
        lbls = np.loadtxt('./fMRIdata/timingLabels_WM_scan1.csv', dtype=np.int32)
        data = []
        labels = []
        for i in range(1,31):
            tmp_data = pd.read_csv(f'./fMRIdata/subject_1{i:02d}_scan{scan_id}.csv')
            tmp_data = sliding_window(tmp_data.values, ws)
            tmp_lbls = get_sliding_labels(lbls, ws)
            data.append(tmp_data)
            labels.append(tmp_lbls)
        self.data = np.stack(data)
        self.labels = np.stack(labels)[..., None] - 1.

        if shuffle:
            random.shuffle(self.data)

        self.nSamples = len(self.data) 
        print(self.nSamples)
        self.nClasses = 17
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        return {'data': torch.from_numpy(self.data[idx].astype(np.float32)), 
                'label': torch.from_numpy(self.labels[idx].astype(np.compat.long))}


class SPDNet(nn.Module):
    def __init__(self, hs):
        super(SPDNet, self).__init__()
        self.trans1 = SPDTransform(57, 20)
        self.trans2 = SPDTransform(20, 10)
        self.trans3 = SPDTransform(10, hs)
        self.rect1  = SPDRectified()
        self.rect2  = SPDRectified()
        self.rect3  = SPDRectified()
        self.tangent = SPDTangentSpace(hs)
        self.linear = nn.Linear(int(hs*(hs+1)/2), 17, bias=True)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.trans1(x)
        x = self.rect1(x)
        x = self.trans2(x)
        x = self.rect2(x)
        x = self.trans3(x)
        x = self.rect3(x)
        x = self.tangent(x)
        # x = self.dropout(x)
        x = self.linear(x)
        return x

def sliding_window_spd(x, ws=30):
    d, n = x.shape
    return np.stack([(1/n)*np.matmul(x[:, i:ws+i], x[:, i:ws+i].T) for i in range(n-ws)])

def get_sliding_labels_spd(x, ws=30):
    n = x.shape[0]
    return np.stack([x[int(ws/2)+i] for i in range(n-ws)])

class SPDataset(Dataset):
    def __init__(self, shuffle=False, scan_id=1, ws=30):
        super(SPDataset, self).__init__()
        self.scan_id = scan_id
        self.ws = ws
        lbls = np.loadtxt('./fMRIdata/timingLabels_WM_scan1.csv', dtype=np.int32)
        data = []
        labels = []
        for i in range(1,31):
            tmp_data = pd.read_csv(f'./fMRIdata/subject_1{i:02d}_scan{scan_id}.csv')
            tmp_data = sliding_window_spd(tmp_data.values, ws)
            tmp_lbls = get_sliding_labels_spd(lbls, ws)
            data.append(tmp_data)
            labels.append(tmp_lbls)
        self.data = np.concatenate(data, axis=0)
        self.labels = np.concatenate(labels, axis=0)[..., None] - 1.

        if shuffle:
            random.shuffle(self.data)

        self.nSamples = len(self.data) 
        print(self.nSamples)
        self.nClasses = 17
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        return {'data': torch.from_numpy(self.data[idx].astype(np.float32)), 
                'label': torch.from_numpy(self.labels[idx].astype(np.compat.long))}

def iterate(args, epoch, model, dataset, optimizer, criterion, best_acc, train, exp_id):
    if train:
        model.train()
    else:
        model.eval()
    test_loss = 0
    correct = 0.0
    total = 0.0
    bar = tqdm(enumerate(dataset))
    for batch_idx, sample_batched in bar:
        inputs = sample_batched['data']
        targets = sample_batched['label'].squeeze()

        if args.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        if train:
            optimizer.zero_grad()
        outputs = model(inputs)
        if args.model=='rspd':
            outputs = outputs.view(-1, outputs.shape[-1])
            targets = targets.view(-1)
        loss = criterion(outputs, targets)
        if train:
            loss.backward()
            optimizer.step()

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    if train:
        filename = f'./checkpoint/{args.model}/{exp_id}'
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            #state = {
            #    'net': model,
            #    'acc': acc,
            #    'epoch': epoch,
            #}
            if not os.path.isdir(filename):
                os.makedirs(filename)
            torch.save(model.state_dict(), os.path.join(filename, 'ckpt.pth'))
            best_acc = acc
    return (test_loss/(batch_idx+1), 100.*correct/total)

def isfloat(x):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b

def get_model_url(model, id):
    return os.path.join('./checkpoint', *[model, id])

def load_args(url):
    parser = argparse.ArgumentParser()
    args={}
    args_url = os.path.join(url, 'args.txt')
    if os.path.exists(args_url):
        with open(args_url, 'r') as f:
            ns = f.read()
            for arg in ns[10:].split(','):
                arg = arg.split('=')
                arg[1] = arg[1].strip('\'')
                arg[1] = arg[1].rstrip(')')
                v = arg[1]
                if(arg[1]=='True'):
                    v=True
                if(arg[1]=='False'):
                    v=False
                if(isfloat(arg[1])):
                    v=float(arg[1])
                if(isint(arg[1])):
                    v=int(arg[1])
                args[arg[0].strip()]=v
    return Namespace(**args)


