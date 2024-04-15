import warnings
warnings.filterwarnings('ignore')
import argparse
import random

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import r2_score, explained_variance_score
from torch.utils.tensorboard import SummaryWriter

from model import Net
from representationsDataset import *
from early_stop import EarlyStopping

def main(local_path, state_path, log_path, batch_size=64, split_rate=0.8, lr=0.005, iterations=1000, tps=32, device=torch.device("cpu"), layer=48, nums=None):
    
    dataset = representationsDataset(local_path, layer, nums)
    '''EarlyStopping and SummaryWriter'''
    early_stopping = EarlyStopping(state_path) 
    writer = SummaryWriter(log_path)
    
    train_dataset, test_dataset = split_dataset(dataset, split_rate)
    # train_dataset, _ = get_dataset(train_dataset, 0.1111)
    train_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=train_dataset,
        shuffle=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=test_dataset,
        shuffle=False,
        drop_last=True
    )

    model = Net(16)
    # if ifload == True:
    print('load state dict')
    model.load_state_dict(torch.load(state_path),strict = False)   #添加了false
    model.to(device)
    
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss(reduction='mean')
    scheduler = StepLR(opt, step_size=20, gamma=0.6)

    for iteration in range(iterations):
        tps = 0
        train_loss = 0.0
        train_pcc = 0
        for data, labels in train_dataloader:
            tps += 1
            data = data.to(device)
            labels = labels.to(device)
            predictions = model(data)

            _loss = loss_func(predictions, labels)
            
            opt.zero_grad()
            _loss.backward()
            opt.step()
            
            train_loss += _loss
            train_pcc += np.corrcoef(np.array(predictions.detach().cpu()).squeeze(),
                                    np.array(labels.detach().cpu()).squeeze())[0][1]

        train_pcc /= tps
        train_loss = train_loss / tps
        
        
        tps = 0
        test_loss = 0.0
        test_pcc = 0
        for data, labels in test_dataloader:
            tps += 1
            data = data.to(device)
            labels = labels.to(device)
            
            predictions = model(data)
            _loss = loss_func(predictions, labels)

            test_loss += _loss
            test_pcc += np.corrcoef(np.array(predictions.detach().cpu()).squeeze(),
                                    np.array(labels.detach().cpu()).squeeze())[0][1]
            
        scheduler.step()
        test_pcc /= tps
        test_loss = test_loss / tps
        
        if iteration % 2 == 0:
            print('iteration: {:d} train loss: {:.3f} train pcc: {:.3f} test loss: {:.3f} test pcc: {:.3f}'.format(iteration, train_pcc.item(), train_pcc, test_loss.item(), test_pcc))
        writer.add_scalar('train_pcc', train_pcc, iteration)
        writer.add_scalar('train_iteration_error', train_loss, iteration)
        writer.add_scalar('test_pcc', test_pcc, iteration)
        writer.add_scalar('test_iteration_error', test_loss, iteration)
        # early_stopping(test_iteration_error, model)
        #达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')
    parser.add_argument('--local_path', type = str)
    parser.add_argument('--state_path', type=str)
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='number of batch_size (default: 64)')
    parser.add_argument('--split_rate', type=float, default=0.8, metavar='LR',
                        help='split_rate for dataset (default: 0.8)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--iterations', type=int, default=200, metavar='N',
                        help='number of iterations (default: 100)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=256, metavar='N',
                        help='tasks per step (default: 32)')
    parser.add_argument('--layer', type=int, default=48, metavar='N',
                        help='number of layer (default: 48)')
    parser.add_argument('--nums', type=int, default = 1000)


    
    # parser.add_argument('--maml-lr', type=float, default=0.0005, metavar='LR',
    #                     help='learning rate for MAML (default: 0.01)')
    # parser.add_argument('--ways', type=int, default=1, metavar='N',
    #                     help='number of ways (default: 5)')
    # parser.add_argument('--shots', type=int, default=1, metavar='N',
    #                     help='number of shots (default: 1)')
    # parser.add_argument('-tps', '--tasks-per-step', type=int, default=256, metavar='N',
    #                     help='tasks per step (default: 32)')
    # parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=5, metavar='N',
    #                     help='steps per fast adaption (default: 5)')
    # parser.add_argument('--download-location', type=str, default="/tmp/mnist", metavar='S',
    #                     help='download location for train data (default : /tmp/mnist')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--ifload', type=bool, default=False)
    


    args = parser.parse_args()

    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if use_cuda:
    #     torch.cuda.manual_seed(args.seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # device = torch.device("cuda" if use_cuda else "cpu")
    device = 'cuda'
    
    main(local_path=args.local_path,
         state_path = args.state_path,
         log_path=args.log_path,
         batch_size=args.batch_size,
         split_rate=args.split_rate,
         lr=args.lr,
         iterations=args.iterations,
         tps=args.tasks_per_step,
         device=device,
         layer=args.layer,
         nums = args.nums)

    # main(local_path = , 
    #      state_path, 
    #      log_path, 
    #      batch_size=64, 
    #      split_rate=0.8, 
    #      lr=0.005, 
    #      iterations=1000, 
    #      tps=32, 
    #      device=torch.device("cpu"), 
    #      layer=48, 
    #      nums=None)
    
    
# python fine_tuning.py --local_path '../nn4dms_esm/data' --state_path ./_states/model.pt --log_path ./_logs_fine_tuning
