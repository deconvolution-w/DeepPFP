from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pathlib
import torch
import numpy as np
import random


def standardization(data):
    '''
    Standardize the data
    x = (x - mean(x)) / std(x)
    '''
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return data / max(abs(data))

class representationsDataset(Dataset):
    def __init__(self, local_path, layer, nums=None, test=False):
        '''
        local_path: The path where the data is stored
        layer: The representation vector corresponding to 
               the Nth layer attention mechanism of the ESM model
        nums: The amount of data sampled at the time of the test
        '''
        print('load data')
        '''
        Read the weight file after ESM characterization in this folder
        '''

        files = pathlib.Path(local_path).rglob("*.pt")
        self.filelist = [file for file in files]
        '''
        Random sampling of data
        '''
        if nums:
            self.filelist = random.sample(self.filelist, nums)
    
        self.layer = layer
        
        '''
        Representation vector and its corresponding label
        '''

        # test1 = self.filelist[0]
        # test2 = torch.load(test1)
        # erro1 = test2['mean_representations'][self.layer].float()
        # label1 = float(str(test1).split('/')[-1].split('_')[-1][:-3])

        # self.representations = [torch.load(_)['mean_representations'][self.layer].float() for _ in self.filelist]
        self.labels, self.representations = [], []
        for _ in self.filelist:
            if torch.load(_)['mean_representations'].get(self.layer) == None:
                continue
            self.representations.append(torch.load(_)['mean_representations'].get(self.layer).float())
            self.labels.append(float(str(_).split('/')[-1].split('_')[-1][:-3]))
        self.labels = standardization(self.labels)

    def __getitem__(self, index):
        representations = self.representations[index]
        label = torch.Tensor([self.labels[index]])
        label = torch.Tensor(label)
        return representations, label
    
    def __len__(self):
        return len(self.labels)   
    
def concatdataset(set_list):
    '''
    Concat the datasets together
    '''
    dsab_cat = ConcatDataset(set_list)
    return dsab_cat

def split_dataset(dataset, split_rate):
    '''
    Randomly divide the dataset by a certain percentage
    '''
    full_length = len(dataset)
    train_size = int(split_rate * full_length)
    test_size = full_length - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, valid_dataset

if __name__ == '__main__':
    local_paths = ['../nn4dms_esm/data/avgfp/avgfp_15B_',
                   '../nn4dms_esm/data/bgl3/bgl3_15B_',
                   '../nn4dms_esm/data/gb1/gb1_15B_',
                   '../nn4dms_esm/data/pab1/pab1_15B_',
                   '../nn4dms_esm/data/ube4b/ube4b_15B_',
                #    '/mnt/sda1/wh/nn4dms_esm/data/avgfp/avgfp_15B_',
                #    '/mnt/sda1/wh/nn4dms_esm/data/bgl3/bgl3_15B_',
                #    '/mnt/sda1/wh/nn4dms_esm/data/gb1/gb1_15B_',
                #    '/mnt/sda1/wh/nn4dms_esm/data/pab1/pab1_15B_',
                #    '/mnt/sda1/wh/nn4dms_esm/data/ube4b/ube4b_15B_',
                #    '/mnt/sda1/wh/ToWH_without_padding_1208/overlap_ab'
                   ]
    datasets = []
    for path in local_paths:
        datasets.append(representationsDataset(path, 48, 1000))
    all_dataset = concatdataset(datasets)
    train_loader = DataLoader(
        dataset=all_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    train_iter = iter(train_loader)
    print(train_iter.__len__())
    for t in range(10):
        try:
            x, y = next(train_iter)
        except:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        # print(y)
        
    