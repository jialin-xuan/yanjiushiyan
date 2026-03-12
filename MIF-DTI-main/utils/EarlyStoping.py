# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-03-26 17:04
LastEditTime: 2022-11-23 15:32
LastEditors: MrZQAQ
Description: Offer EarlyStoping function
FilePath: /MCANet/utils/EarlyStoping.py
'''

import numpy as np
import torch


class EarlyStopping:
    """
    早停机制 (Early Stopping)
    功能：如果验证集损失在给定的耐心值 (patience) 内没有改善，则停止训练，防止过拟合。
    """

    def __init__(self, savepath=None, patience=7, verbose=False, delta=0, num_n_fold=0):
        """
        Args:
            patience (int): 上次验证集损失改善后等待多久。默认值: 7
            verbose (bool): 如果为True，则为每次验证损失改善打印一条消息。默认值: False
            delta (float): 监测数量的最小变化，小于此变化被认为没有改善。默认值: 0
            savepath (str): 模型保存路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False
        self.delta = delta
        self.num_n_fold = num_n_fold
        self.savepath = savepath

    def __call__(self, score, model, num_epoch):
        # 这里的score似乎是越高越好（例如Accuracy或AUC），因为逻辑是 score < best_score + delta 时计数增加
        # 如果score是loss，则逻辑应该是相反的。
        # 查看RunModel.py调用: early_stopping(Accuracy_dev, model, epoch)
        # 传入的是Accuracy，所以是越高越好。

        if self.best_score == -np.inf:
            self.save_checkpoint(score, model, num_epoch)
            self.best_score = score

        elif score < self.best_score + self.delta:
            # 如果当前得分没有超过最佳得分（加上delta），则计数器加1
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 如果当前得分超过最佳得分，则保存模型并重置计数器
            self.save_checkpoint(score, model, num_epoch)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model, num_epoch):
        '''当验证集得分提升时保存模型。'''
        if self.verbose:
            print(
                f'Have a new best checkpoint: ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.savepath +
                   f'/valid_best_checkpoint-{str(model.device).replace(":", "_")}.pth')
