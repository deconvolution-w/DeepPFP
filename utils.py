import torch
import numpy as np
def get_loss_pcc(data, label, model, loss_func):
    '''
    Calculate the loss and the Pearson correlation coefficient.
    
    data: the data input to the model
    label: the label corresponding to the data
    model: the model used for the experiment
    loss_func: loss function
    '''
    predictions = model(data)
    loss = loss_func(predictions, label)
    pcc = np.corrcoef(np.array(predictions.detach().cpu()).squeeze(),
                    np.array(label.detach().cpu()).squeeze())[0][1]
    
    # #新加
    # device = "cuda:1"
    # loss.to(device)
    # pcc.to(device)
    return loss, pcc

def test_process(test_iterset, testset_max, test_datasets, model, loss_func, device='cpu'):
    '''
    Test process
    
    test_iterset: the list of test dataset's iter
    testset_max: the largest dataset's length
    test_datasets: the list of test dataset
    model: the model used for the experiment
    device: use cpu or cuda to deal with the dataset
    '''
    test_loss = 0
    test_pcc = 0
    pccs = [[] for i in range(len(test_iterset))]
    
    for tps in range(int(testset_max / 5)):
        loss = 0
        pcc = 0
        for iter_num in range(len(test_iterset)):
            try:
                data, label = next(test_iterset[iter_num])
            except:
                test_iterset[iter_num] = iter(test_datasets[iter_num])
                data, label = next(test_iterset[iter_num])
            data, label = data.to(device), label.to(device)

            _loss, _pcc = get_loss_pcc(data, label, model, loss_func)
            loss += _loss
            pccs[iter_num].append(_pcc)
            pcc += _pcc
        test_pcc += (pcc / len(test_iterset))
        test_loss += (loss / len(test_iterset))
    
    test_loss /= tps
    test_pcc /= tps
    
    return test_loss, test_pcc, pccs
    
