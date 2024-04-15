## rebuild finished
## 翻译 finish
'''
以数据集混合的方式在多个数据集上进行预训练
多个训练数据集会混合后随机打乱
Pre-training on multiple datasets with add dataset
Multiple training data sets are mixed and then randomly disrupted
'''
import argparse
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter   
from sklearn.metrics import r2_score, explained_variance_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
np.seterr(divide='ignore',invalid='ignore')


from early_stop import EarlyStopping
from model import Net
from representationsDataset import *
from utils import *

def main(local_paths, log_path, model_save_path, batch_size=64, split_rate=0.8, lr=0.0001, iterations=1000, device='cpu', nums=None, layer=48):
      
    early_stopping = EarlyStopping(model_save_path) 
    writer = SummaryWriter(log_path)
    '''
    数据集制作
    训练集混合，测试集分别读取（后面会分别计算每一类的皮尔逊相关系数）
    Dataset:train_dataset is mixed and the test_dataset is read separately.
    Pearson correlation coefficients for each class will be calculated separately later.
    '''
    train_datasets = []
    test_datasets = []
    testset_max = 0

    for path in local_paths:
        _dataset = representationsDataset(path, layer, nums)
        train_dataset, test_dataset = split_dataset(_dataset, split_rate)
        test_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=test_dataset,
        shuffle=True,
        drop_last=True
        )
        test_datasets.append(test_dataloader)
        train_datasets.append(train_dataset)

        testset_max = test_dataloader.__len__() if test_dataloader.__len__() > testset_max else testset_max
        
    train_datasets = concatdataset(train_datasets)
    train_dataloader =DataLoader(
        batch_size=batch_size,
        dataset=train_datasets,
        shuffle=True,
        drop_last=True
        )

    
    train_iterset = iter(train_dataloader)
    test_iterset = [iter(_) for _ in test_datasets]
    
    '''
    定义模型
    Define models
    '''
    model = Net(16)
    model = model.to(device)

    '''
    定义优化器，损失函数，学习率更新策略
    Define optimizer, loss function, learning rate update strategy
    '''
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=20, gamma=0.6)
    loss_func = nn.MSELoss(reduction='mean')
    

    '''
    训练过程
    Training process
    '''
    for iteration in range(iterations):
        train_loss = 0
        train_pcc = 0
        for tps in range(int(train_dataloader.__len__())):
            # loss = 0
            # pcc = 0
            try:
                data, label = next(train_iterset)
            except:
                train_iterset = iter(train_dataloader)
                data, label = next(train_iterset)
                
            '''
            混合后的数据集每次计算损失后不会立即进行反向传播
            The blended dataset will not backward immediately after each computed loss
            此处对应maml的反向传播方式, 在计算完所有损失后再进行反向传播
            Corresponding to Model-Agnostic Meta-Learning's backpropagation method, 
            where backpropagation is performed after all losses have been calculated
            '''
            data, label = data.to(device), label.to(device)
            _loss, _pcc = get_loss_pcc(data, label, model, loss_func)
            # loss += _loss
            # pcc += _pcc
            '''反向传播过程
            backward process'''
            opt.zero_grad()
            _loss.backward()
            opt.step()
            
            train_pcc += _pcc
            train_loss += _loss

        train_loss /= tps
        train_pcc /= tps
        
        '''
        测试过程
        Test process
        '''
     
        test_loss, test_pcc, pccs = test_process(test_iterset, testset_max, test_datasets, model, loss_func, device)
        
        scheduler.step()
        
        '''输出损失和皮尔逊相关系数
        Output loss and Pearson correlation coefficient'''
        print('iteration : {:d} train_loss : {:.3f} train_pcc : {:.3f} test_loss : {:.3f} test_pcc : {:.3f}'.format(iteration, train_loss.item(), train_pcc, test_loss.item(), test_pcc))
        writer.add_scalar('train_pcc', train_pcc, iteration)
        writer.add_scalar('train_loss', train_loss, iteration)
        writer.add_scalar('test_pcc', test_pcc, iteration)
        writer.add_scalar('test_loss', test_loss, iteration)
        for i in range(len(test_iterset)):
            # writer.add_scalar('test_pcc' + str(i), sum(pccs[i]) / len(pccs[i]), iteration)
            writer.add_scalar('test_pcc_' + local_paths[i].split('/')[-1][:-6], sum(pccs[i]) / len(pccs[i]), iteration)

        
        writer.add_scalar('lr', opt.state_dict()['param_groups'][0]['lr'], iteration)

        early_stopping(test_loss, model)
        '''
        达到早停止条件时, early_stop会被置为True
        early_stop will be set to True when the early stop condition is reached
        '''
        if early_stopping.early_stop:
            print("Early stopping")
            break



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Pretrain_add_dataset')
    parser.add_argument('--local_paths', nargs='+', help='local path: where the data is.', required=True)

    parser.add_argument('--logpath', type=str)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='number of batch_size (default: 64)')
    parser.add_argument('--split_rate', type=float, default=0.8, metavar='LR',
                        help='split_rate for dataset (default: 0.8)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                        help='number of iterations (default: 100)')
    parser.add_argument('--layer', type=int, default=48, metavar='N',
                        help='number of layer (default: 48)')
    

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    args = parser.parse_args()

    # use_cuda = not args.no_cuda and torch.cuda.is_available() -------------------------------------------
    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if use_cuda:
    #     torch.cuda.manual_seed(args.seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # device = torch.device("cuda:1" if use_cuda else "cpu")
    # device = torch.device("cuda:1")
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

#运行
# python pretrain_add_dataset.py --local_path '../nn4dms_esm/data/avgfp/avgfp_15B_' '../nn4dms_esm/data/bgl3/bgl3_15B_' '../nn4dms_esm/data/gb1/gb1_15B_' '../nn4dms_esm/data/pab1/pab1_15B_' '../nn4dms_esm/data/ube4b/ube4b_15B_' --logpath ./logs --model_save_path ./_model_save/_pad_model

