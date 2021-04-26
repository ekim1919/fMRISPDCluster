import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import *
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
from spdnet.optimizer import StiefelMetaOptimizer

use_cuda = True
BATCH_SIZE = 7
WS = 30 #window_size

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
    def __init__(self):
        super(input_layer, self).__init__()
        self.trans1 = SPDTransform(57, 20)
        self.trans2 = SPDTransform(20, 10)
        self.trans3 = SPDTransform(10, 2)
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
    def __init__(self):
        super(output_layer, self).__init__()
        self.tangent = SPDTangentSpace(2)
        self.linear = nn.Linear(3, 17, bias=True)

    def forward(self, x):
        x = self.tangent(x)
        x = self.linear(x)
        return x

class SPDRNN(nn.Module):
    def __init__(self):
        super(SPDRNN, self).__init__()
        self.input_layer = input_layer()
        self.output_layer = output_layer()
        self.recurrent_layer = recurrent_layer()
        self.alpha = .7

    def forward(self, input, future=0):
        outputs = []
        # The two corresponds to the dimension of the recurrent SPD matrix.
        h_t = torch.zeros(input.size(0), 2, 2, dtype=torch.float)
        if use_cuda:
            h_t = h_t.cuda()

        for i in range(input.size(1)):
            x = input[:, i, :, :]
            tmp_a = self.alpha*self.input_layer(x)
            tmp_b = (1-self.alpha)*self.recurrent_layer(h_t)
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

transformed_dataset = SPDRNNDataset(scan_id=1, ws=WS)
dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

transformed_dataset_val = SPDRNNDataset(scan_id=2, ws=WS)
dataloader_val = DataLoader(transformed_dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

use_cuda = False
model = SPDRNN()
if use_cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = StiefelMetaOptimizer(optimizer)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0.0
    total = 0.0
    bar = tqdm(enumerate(dataloader))
    for batch_idx, sample_batched in bar:
        inputs = sample_batched['data']
        targets = sample_batched['label'].squeeze()

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1.0), 100.*correct/total, correct, total))

    return (train_loss/(batch_idx+1), 100.*correct/total)

best_acc = 0
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0.0
    total = 0.0
    bar = tqdm(enumerate(dataloader_val))
    for batch_idx, sample_batched in bar:
        inputs = sample_batched['data']
        targets = sample_batched['label'].squeeze()

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.view(-1)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/rspd_ckpt.t7')
        best_acc = acc

    return (test_loss/(batch_idx+1), 100.*correct/total)

log_file = open('log_rnn.txt', 'a')

start_epoch = 1
for epoch in range(start_epoch, start_epoch+100):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)

    log_file.write('%d,%f,%f,%f,%f\n' % (epoch, train_loss, train_acc, test_loss, test_acc))
    log_file.flush()

log_file.close()
