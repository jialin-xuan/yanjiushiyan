# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-03-29 13:59
LastEditTime: 2022-11-23 15:33
LastEditors: MrZQAQ
Description: Prepare Data for main process
FilePath: /MCANet/utils/DataPrepare.py
CopyRight 2022 by MrZQAQ. All rights reserved.
'''

import numpy as np

def get_kfold_data(i, datasets, k=5):
    """
    获取K折交叉验证的训练集和验证集
    参数:
        i (int): 当前折数 (0 到 k-1)
        datasets (list): 完整数据集列表
        k (int): 总折数
    返回:
        trainset (list): 训练集
        validset (list): 验证集
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
    打乱数据集
    参数:
        dataset (list): 数据集列表
        seed (int): 随机种子，保证打乱结果可复现
    返回:
        dataset (list): 打乱后的数据集
    """
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
