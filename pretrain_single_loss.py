## rebuild finished
'''
以single loss的方式在多个数据集上进行预训练
多个训练数据集会混合后随机打乱
Pre-training on multiple datasets with single loss
Multiple training datasets are mixed and then randomly disrupted
'''
import warnings
warnings.filterwarnings('ignore')
import argparse
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter   
from sklearn.metrics import r2_score, explained_variance_score


from early_stop import EarlyStopping
from model import Net
from representationsDataset import *
from utils import *

def main(local_paths, log_path, model_save_path, batch_size=32, split_rate=0.8, lr=0.005, iterations=1000, device=torch.device("cpu"), nums=None, layer=48):


    early_stopping = EarlyStopping(model_save_path) 
    writer = SummaryWriter(log_path)

    '''
    数据集制作
    训练集和测试集都是分别读取（后面会分别计算每一类的皮尔逊相关系数）
    Dataset Production
    Both the training and test sets are read separately, and
    Pearson correlation coefficients for each class will be calculated separately later
    '''
    train_datasets = []
    test_datasets = []
    trainset_max = 0
    testset_max = 0
    for path in local_paths:
        _dataset = representationsDataset(path, layer, nums)
        train_dataset, test_dataset = split_dataset(_dataset, split_rate)
        train_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=train_dataset,
        shuffle=True,
        drop_last=True
        )
        test_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=test_dataset,
        shuffle=True,
        drop_last=True
        )
        trainset_max = train_dataloader.__len__() if train_dataloader.__len__() > trainset_max else trainset_max
        testset_max = test_dataloader.__len__() if test_dataloader.__len__() > testset_max else testset_max
        train_datasets.append(train_dataloader)
        test_datasets.append(test_dataloader)

    model = Net(16)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=20, gamma=0.6)
    loss_func = nn.MSELoss(reduction='mean')

    train_iterset = [iter(_) for _ in train_datasets]
    test_iterset = [iter(_) for _ in test_datasets]
    for iteration in range(iterations):
        train_loss = 0
        train_pcc = 0
        for tps in range(int(trainset_max / 5)):
            # loss = []
            loss = 0
            pcc = 0
            for iter_num in range(len(train_iterset)):
                try:
                    data, label = next(train_iterset[iter_num])
                except:
                    train_iterset[iter_num] = iter(train_datasets[iter_num])
                    data, label = next(train_iterset[iter_num])
         
                data, label = data.to(device), label.to(device)
                _loss, _pcc = get_loss_pcc(data, label, model, loss_func)
                
                '''
                每计算一次损失，就会进行一次反向传播
                在每一个循环内，模型都会进行N次反向传播
                Each time a loss is calculated, a back propagation is performed
                Within each loop, the model will backward N times
                '''
                opt.zero_grad()
                _loss.backward()
                opt.step()
                
                # loss.append(_loss)
                loss += _loss
                if _pcc != np.nan:
                    pcc += _pcc

            # loss = sum(loss)
            
            train_pcc += (pcc / len(local_paths))
            train_loss += (loss / len(local_paths))

        train_loss /= tps
        train_pcc /= tps
        
        '''
        测试过程
        Test process
        '''
        test_loss, test_pcc, pccs = test_process(test_iterset, testset_max, test_datasets, model, loss_func, device)

        
        scheduler.step()
        
        print('iteration : {:d} train_loss : {:.3f} train_pcc : {:.3f} test_loss : {:.3f} test_pcc : {:.3f}'.format(iteration, train_loss.item(), train_pcc, test_loss.item(), test_pcc))
        writer.add_scalar('train_pcc', train_pcc, iteration)
        writer.add_scalar('train_loss', train_loss, iteration)
        writer.add_scalar('test_pcc', test_pcc, iteration)
        writer.add_scalar('test_loss', test_loss, iteration)
        for i in range(len(test_iterset)):
            writer.add_scalar('test_pcc_' + local_paths[i].split('/')[-1][:-6], sum(pccs[i]) / len(pccs[i]), iteration)
        writer.add_scalar('lr', opt.state_dict()['param_groups'][0]['lr'], iteration)
        
        
        early_stopping(test_loss, model)
        #达到早停止条件时，early_stop会被置为True
        #Early_stop will be set to True when the early stop condition is reached
        if early_stopping.early_stop:
            print("Early stopping")
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain Single Loss')
    parser.add_argument('--local_paths', nargs='+', help='local path: where the data is.', required=True)
    parser.add_argument('--logpath', type=str)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='number of batch_size (default: 32)')
    parser.add_argument('--split_rate', type=float, default=0.8, metavar='LR',
                        help='split_rate for dataset (default: 0.8)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                        help='number of iterations (default: 1000)')
    parser.add_argument('--layer', type=int, default=48, metavar='N',
                        help='number of layer (default: 48)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()

    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if use_cuda:
    #     torch.cuda.manual_seed(args.seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # device = torch.device("cuda" if use_cuda else "cpu")
    device = 'cuda'

    main (local_paths = args.local_paths,
         log_path = args.logpath,
         model_save_path = args.model_save_path, 
         batch_size = args.batch_size, 
         split_rate = args.split_rate, 
         lr = args.lr, 
         iterations = args.iterations, 
         device = device,
         nums = None, 
         layer = args.layer)

# 运行
# python pretrain_single_loss.py --local_path '../nn4dms_esm/data/avgfp/avgfp_15B_' '../nn4dms_esm/data/bgl3/bgl3_15B_' '../nn4dms_esm/data/gb1/gb1_15B_' '../nn4dms_esm/data/pab1/pab1_15B_' '../nn4dms_esm/data/ube4b/ube4b_15B_' --logpath ./logs3 --model_save_path ./_model_save/_psl_model
