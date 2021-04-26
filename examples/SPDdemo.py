import argparse
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from dataset import *
from utils import *
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
from spdnet.optimizer import StiefelMetaOptimizer

def main(args):
    exp_id = int(time.time())
    
    if args.model == 'spd':
        model = SPDNet(args.hs)
        train_data = SPDataset(scan_id=1, ws=args.ws)
        test_data = SPDataset(scan_id=2, ws=args.ws)
    elif args.model == 'rspd':
        model = RSPDNet(args.hs, args.use_cuda)
        train_data = SPDRNNDataset(scan_id=1, ws=args.ws)
        test_data = SPDRNNDataset(scan_id=2, ws=args.ws)
    if args.use_cuda:
        model = model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=args.bs, 
            shuffle=False, num_workers=4)
    test_data_loader = DataLoader(test_data, batch_size=args.bs, 
            shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer = StiefelMetaOptimizer(optimizer)

    best_acc = 0

    log_file = open('./log.txt', 'a')

    for epoch in range(args.epochs):
        train_loss, train_acc = iterate(args, epoch, model, train_data_loader, 
                                        optimizer, criterion, best_acc, True, exp_id)
        test_loss, test_acc = iterate(args, epoch, model, test_data_loader, 
                                        optimizer, criterion, best_acc, False, exp_id)

        log_file.write('%s,%d,%d,%f,%f,%f,%f\n' % (args.model, exp_id, epoch, train_loss, 
            train_acc, test_loss, test_acc))
        log_file.flush()

    log_file.close()

    arg_file = os.path.join(f'./checkpoint/{args.model}/{exp_id}', 'args.txt')
    with open(arg_file, "w+") as f:
        f.write(str(args))
    
    print(f'logpath: ./checkpoint/{args.model}/{exp_id}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI Options',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-model", default="spd",
                help="Type of model: spd, rspd.")
    parser.add_argument("-epochs", default=7, type=int,
                help="Number of epochs to train for.")
    parser.add_argument("-bs", default=7, type=int,
                help="Batch size.")
    parser.add_argument("-ws", default=30, type=int,
                help="Window size.")
    parser.add_argument("-hs", default=2, type=int,
                help="Smallest reduction in the SPD manifold.")
    parser.add_argument("-lr", default=.01, type=float,
                help="Learning rate.")
    parser.add_argument("-use_cuda", action='store_true',
                help="Flag to use cuda.")
    args = parser.parse_args()

    main(args)
