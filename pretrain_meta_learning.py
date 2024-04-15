## rebuild finished
## 翻译 finish
'''
以meta learning的方式在多个数据集上进行预训练
多个训练数据集会混合后随机打乱
Pre-training on multiple datasets with meta learning
Multiple training datasets are mixed and then randomly disrupted
'''
import argparse
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter   
import learn2learn as l2l
from sklearn.metrics import r2_score, explained_variance_score


from early_stop import EarlyStopping
from model import Net
from representationsDataset import *
from utils import *



def main(local_paths, log_path, model_save_path, batch_size=32, split_rate=0.8, lr=0.005, maml_lr=0.01, iterations=1000, fas=5, device=torch.device("cpu"),
         nums=None, layer=48):

    # EarlyStopping and SummaryWriter
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
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataloader)
        testset_max = test_dataloader.__len__() if test_dataloader.__len__() > testset_max else testset_max

    train_dataset = concatdataset(train_datasets)
    train_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=train_dataset,
        shuffle=True,
        drop_last=True
    )
    train_iterset = iter(train_dataloader)
    test_iterset = [iter(_) for _ in test_datasets]
    '''
    定义模型并将他们放在maml框架下
    Define models and place them in the Model-agnostic meta-learning framework
    '''

    model = Net(16)
    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr, first_order=False, allow_unused=True)
    
    '''
    定义优化器，损失函数，学习率更新策略
    Define optimizer, loss function, learning rate update strategy
    '''
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    loss_func = nn.MSELoss(reduction='mean')
    scheduler = StepLR(opt, step_size=5, gamma=0.8)

    '''
    训练过程
    Training process
    '''
    for iteration in range(iterations):
        train_loss = 0.0
        train_pcc = 0
        for tps in range(int(train_dataloader.__len__())):
            learner = meta_model.clone()
            try:
                # 当迭代器内还有数据时直接读取
                # Read directly when there is still data in the iterator
                data, labels = next(train_iterset)
            except:
                # 当迭代器内没有数据时，更新迭代器
                # Update the iterator when there is no data in iterator
                train_iterset = iter(train_dataloader)
                data, labels = next(train_iterset)
            data, labels = data.to(device), labels.to(device)
            adaptation_data, adaptation_labels = data[:int(len(data) / 2)], labels[:int(len(data) / 2)]
            evaluation_data, evaluation_labels = data[int(len(data) / 2):], labels[int(len(data) / 2):]
            # 更新N次网络参数
            # Update the network parameters N times
            for _ in range(fas):   
                train_error = loss_func(learner(adaptation_data), adaptation_labels)
                learner.adapt(train_error) 
            
            evaluation_data, evaluation_labels = evaluation_data.to(device), evaluation_labels.to(device)

            # Compute validation loss
            predictions = learner(evaluation_data)

            valid_error = loss_func(predictions, evaluation_labels)

            train_loss += valid_error
            train_pcc += np.corrcoef(np.array(predictions.detach().cpu()).squeeze(),
                        np.array(evaluation_labels.detach().cpu()).squeeze())[0][1]

        train_loss /= tps
        train_pcc /= tps

        # Take the meta-learning step
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        
        
        '''
        测试过程
        Test process
        '''
        # N个数据集 and N个皮尔逊相关系数
        #N datasets and N Pearson correlation coefficients
        test_loss, test_pcc, pccs = test_process(test_iterset, testset_max, test_datasets, model, loss_func, device)

        
        scheduler.step()
        
        # 输出损失和皮尔逊相关系数
        # Output loss and Pearson correlation coefficient
        print('iteration : {:d} train_loss : {:.3f} train_pcc : {:.3f} test_loss : {:.3f} test_pcc : {:.3f}'.format(iteration, train_loss.item(), train_pcc, test_loss.item(), test_pcc))
        
        # 使用tensorboard记录损失、皮尔逊相关系数和学习率的变化
        # using tensorboard to Record changes in loss, Pearson correlation coefficient and learning rate
        writer.add_scalar('train_pcc', train_pcc, iteration)
        writer.add_scalar('train_loss', train_loss, iteration)
        writer.add_scalar('test_pcc', test_pcc, iteration)
        writer.add_scalar('test_loss', test_loss, iteration)
        for i in range(len(test_iterset)):
            # writer.add_scalar('test_pcc' + str(i), sum(pccs[i]) / len(pccs[i]), iteration)
            writer.add_scalar('test_pcc_' + local_paths[i].split('/')[-1][:-6], sum(pccs[i]) / len(pccs[i]), iteration)

        writer.add_scalar('lr', opt.state_dict()['param_groups'][0]['lr'], iteration)
        
        '''
        达到早停止条件时，early_stop会被置为True
        此时训练停止，保存模型参数
        early_stop will be set to True when the early stop condition is reached
        the training stops and the model parameters are saved
        '''
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='pretrain_meta_learning')
    parser.add_argument('--local_paths', nargs='+', help='local path: where the data is.', required=True),
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='number of batch_size (default: 32)')
    parser.add_argument('--split_rate', type=float, default=0.8, metavar='LR',
                        help='split_rate for dataset (default: 0.8)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--maml_lr', type=float, default=0.001, metavar='LR',
                        help='maml learning rate (default: 0.01)')
    parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                        help='number of iterations (default: 1000)')
    parser.add_argument('--fas', type=int, default=5, metavar='N')
    parser.add_argument('--layer', type=int, default=48, metavar='N',
                        help='number of layer (default: 48)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--nums', type=int, default=9999999, metavar='N',
                        help='length of every dataset')
    args = parser.parse_args()

    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if use_cuda:
    #     torch.cuda.manual_seed(args.seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    device = 'cuda'

    main (local_paths = args.local_paths,
         log_path = args.log_path,
         model_save_path = args.model_save_path, 
         batch_size = args.batch_size, 
         split_rate = args.split_rate, 
         lr = args.lr, 
         maml_lr=args.maml_lr,
         iterations = args.iterations,
         fas=args.fas,
         device = device,
         nums = args.nums, 
         layer = args.layer)
#运行
# python pretrain_meta_learning.py --local_path '../nn4dms_esm/data/avgfp/avgfp_15B_' '../nn4dms_esm/data/bgl3/bgl3_15B_' '../nn4dms_esm/data/gb1/gb1_15B_' '../nn4dms_esm/data/pab1/pab1_15B_' '../nn4dms_esm/data/ube4b/ube4b_15B_' --log_path ./logs2 --model_save_path ./_model_save/_pml_model