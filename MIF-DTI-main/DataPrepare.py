# -*- coding:utf-8 -*-

import numpy as np

def get_kfold_data(i, datasets, k=5):
    """
    获取K折交叉验证的训练集和验证集
    参数:
        i (int): 当前折数
        datasets (list): 数据集
        k (int): 总折数
    """
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset

def shuffle_dataset(dataset, seed):
    """
    随机打乱数据集
    """
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
