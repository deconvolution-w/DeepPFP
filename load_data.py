import warnings
warnings.filterwarnings('ignore')
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


import os
def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
 
list_name=[]
path='/mnt/sda1/wh/metalearning_rebuild/avgfp_logs/'   #文件夹路径
listdir(path,list_name)

for path in list_name:
    # path = '/mnt/sda1/wh/metalearning/code/lossadd/class-4-add-datasetavgfp_15B_/events.out.tfevents.1677830243.lab312-Super-Server.1895904.0'
    save_name = path.split('/')[-2]
    print(save_name)
    #加载日志数据
    ea=event_accumulator.EventAccumulator(path) 
    ea.Reload()
    keys = ea.scalars.Keys()

    data = pd.DataFrame(columns=keys)

    for _ in keys:
        val_psnr=ea.scalars.Items(_)
        try:
            __ = [i.value for i in val_psnr]
            data[_] = __
        except:
            for i in range(len(val_psnr)):
                data[_][i] = val_psnr[i].value 
    
    if save_name.startswith('class-4'):
        if 'avgfp' in save_name:
            data = data.rename(columns={'test_pcc0': 'test_bgl3', 'test_pcc1': 'test_gb1', 'test_pcc2': 'test_pab1', 'test_pcc3': 'test_ube4b'})
        if 'bgl3' in save_name:
            data = data.rename(columns={'test_pcc0': 'test_avgfp', 'test_pcc1': 'test_gb1', 'test_pcc2': 'test_pab1', 'test_pcc3': 'test_ube4b'})
        if 'gb1' in save_name:
            data = data.rename(columns={'test_pcc0': 'test_avgfp', 'test_pcc1': 'test_bgl3', 'test_pcc2': 'test_pab1', 'test_pcc3': 'test_ube4b'})
        if 'pab1' in save_name:
            data = data.rename(columns={'test_pcc0': 'test_avgfp', 'test_pcc1': 'test_bgl3', 'test_pcc2': 'test_gb1', 'test_pcc3': 'test_ube4b'})
        if 'ube4b' in save_name:
            data = data.rename(columns={'test_pcc0': 'test_avgfp', 'test_pcc1': 'test_bgl3', 'test_pcc2': 'test_gb1', 'test_pcc3': 'test_pab1'})
    if save_name.startswith('class-5') or save_name.startswith('maml-5'):
        data = data.rename(columns={'test_pcc0': 'test_avgfp', 'test_pcc1': 'test_bgl3', 'test_pcc2': 'test_gb1', 'test_pcc3': 'test_pab1', 'test_pcc4': 'test_ube4b'})
        
    print(data)
    
    
    
    
    data.to_csv('plot_data/' + save_name + '.csv')
    
    
    
    
    
