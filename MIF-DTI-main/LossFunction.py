# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyLoss(nn.Module):
    """
    PolyLoss损失函数
    功能：在CrossEntropyLoss的基础上引入多项式展开项，通过调整epsilon参数来调节对不同概率预测的关注程度。
    公式: L_poly = L_CE + epsilon * (1 - pt)
    """
    def __init__(self, weight_loss, DEVICE, epsilon=1.0):
        super(PolyLoss, self).__init__()
        # 基础损失为CrossEntropyLoss
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        # 创建One-hot编码
        one_hot = torch.zeros((labels.shape[0], 2), device=self.DEVICE).scatter_(
            1, torch.unsqueeze(labels, dim=-1), 1)
        # 计算预测概率 pt
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        # 计算CE Loss
        ce = self.CELoss(predicted, labels)
        # 计算PolyLoss
        poly1 = ce + self.epsilon * (1-pt)
        return torch.mean(poly1)


class CELoss(nn.Module):
    """
    交叉熵损失函数 (Cross Entropy Loss)
    功能：标准的分类损失函数。
    """
    def __init__(self, weight_CE, DEVICE):
        super(CELoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_CE)
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        return self.CELoss(predicted, labels)
