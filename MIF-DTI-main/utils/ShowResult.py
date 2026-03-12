# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-11-23 13:40
LastEditTime: 2022-11-23 15:29
LastEditors: MrZQAQ
Description: Show and save result
FilePath: /MCANet/utils/ShowResult.py
'''

import numpy as np


def show_result(DATASET, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List, Ensemble=False):
    """
    显示和保存训练结果
    参数:
        DATASET: 数据集名称
        Accuracy_List: 准确率列表 (每折)
        Precision_List: 精确率列表 (每折)
        Recall_List: 召回率列表 (每折)
        AUC_List: AUC列表 (每折)
        AUPR_List: AUPR列表 (每折)
        Ensemble: 是否为集成模型结果
    """
    # 计算各项指标的均值和方差
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(
        Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)

    if Ensemble == False:
        print("The model's results:")
        filepath = "./{}/results.txt".format(DATASET)
    else:
        print("The ensemble model's results:")
        filepath = "./{}/ensemble_results.txt".format(DATASET)
    
    # 写入文件
    with open(filepath, 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(
            Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(
            Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(
            Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')
    
    # 打印到控制台
    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(
        Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))
