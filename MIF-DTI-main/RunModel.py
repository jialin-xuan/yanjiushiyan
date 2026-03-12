# -*- coding:utf-8 -*-

import os
import random
import joblib
import pandas as pd
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import hyperparameter
from model import MIFDTI
from utils.DataPrepare import get_kfold_data, shuffle_dataset
from utils.DataSetsFunction import CustomDataSet, collate_fn
from utils.EarlyStoping import EarlyStopping
from LossFunction import CELoss, PolyLoss
from utils.TestModel import test_model
from utils.ShowResult import show_result
from utils import protein_init, ligand_init, ProteinMoleculeDataset
import torch_geometric.loader as pyg_loader
from utils.sensitivity_analyzer import SensitivityAnalyzer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from datetime import datetime

def run_MIF_model(SEED, DATASET, MODEL, K_Fold, LOSS, device):
    '''设置随机种子，保证实验可复现'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''初始化超参数'''
    hp = hyperparameter()

    '''从文本文件加载数据集'''
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    
    print("Train in " + DATASET)
    print("load data")
    # 读取数据集文件
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''设置损失函数权重，针对不同数据集设置类别权重以处理不平衡问题'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    # BD2D数据集的特殊处理（可能是因为数据集较大或有特定的分割）
    if DATASET == "BD2D":
        split_pos = 52010
        train_data_list = data_list[:split_pos]
        test_data_list = data_list[split_pos:]
        '''打乱数据'''
        print("data shuffle")
        train_data_list = shuffle_dataset(train_data_list, SEED)
    else:
        '''打乱数据'''
        print("data shuffle")
        data_list = shuffle_dataset(data_list, SEED)

        '''将数据集分割为训练集&验证集和测试集 (80% 训练验证, 20% 测试)'''
        split_pos = len(data_list) - int(len(data_list) * 0.2)
        train_data_list = data_list[0:split_pos]
        test_data_list = data_list[split_pos:-1]
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))

    '''数据预处理与加载'''
    # 加载或生成蛋白质图数据
    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein-new.pkl'
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    # 加载或生成配体（药物）图数据
    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi-new.pkl'
    if os.path.exists(ligand_path):
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    torch.cuda.empty_cache()

    '''初始化评价指标列表'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    # 进行K折交叉验证
    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        # 获取当前折的训练集和验证集数据
        # 注意：i_fold 是从 0 开始的，get_kfold_data 也接受 0-based index
        train_dataset, valid_dataset = get_kfold_data(i_fold, train_data_list, k=K_Fold)
        
        # 调试信息：打印数据集大小
        print(f"Fold {i_fold+1}: Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}")
        
        if len(train_dataset) == 0:
            print("Warning: Train dataset is empty! Skipping this fold.")
            continue

        # 构建PyTorch Dataset对象
        train_dataset = ProteinMoleculeDataset(train_dataset, ligand_dict, protein_dict, device=device)
        valid_dataset = ProteinMoleculeDataset(valid_dataset, ligand_dict, protein_dict, device=device)
        test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=device)
        train_size = len(train_dataset)

        # 构建DataLoader，用于批量加载数据
        # follow_batch参数用于PyG DataBatch，指示哪些属性需要特殊的batch处理
        train_loader = pyg_loader.DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
        valid_loader = pyg_loader.DataLoader(valid_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
        test_loader = pyg_loader.DataLoader(test_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
                                    
        """ create model"""
        # 实例化模型
        model = MODEL(device=device)

        """Initialize weights"""
        # 初始化模型权重
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        """create optimizer and scheduler"""
        # 创建优化器 (AdamW)
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

        # 创建学习率调度器 (CyclicLR)
        # 确保 max_lr 为 Learning_rate * 10 且 cycle_momentum=False
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)
        # if LOSS == 'PolyLoss':
        #     Loss = PolyLoss(weight_loss=weight_loss,
        #                     DEVICE=device, epsilon=hp.loss_epsilon)
        # else:
        # 定义损失函数 (CrossEntropyLoss)
        Loss = CELoss(weight_CE=weight_loss, DEVICE=device)


        """Output files"""
        # 设置结果保存路径
        save_path = "./" + DATASET + "/{}".format(i_fold+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

        # 保存超参数到JSON文件
        hp_dict = {k: v for k, v in hp.__dict__.items() if not k.startswith('__')}
        with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as f:
            json.dump(hp_dict, f, indent=4)

        # Initialize CSV file with pandas
        # 初始化CSV文件记录训练过程中的指标
        csv_file_path = save_path + '/training_results.csv'
        results_df = pd.DataFrame(columns=['Epoch', 'Train_Loss', 'Valid_Loss', 'Valid_AUC', 'Valid_PRC', 'Valid_Accuracy', 'Valid_Precision', 'Valid_Recall'])
        results_df.to_csv(csv_file_path, index=False)
        
        # --- 新增：日志目录 ---
        training_logs_dir = 'results/training_logs/'
        if not os.path.exists(training_logs_dir):
            os.makedirs(training_logs_dir)
        
        # 记录特征敏感度日志
        sensitivity_log_path = os.path.join(training_logs_dir, f'{DATASET}_fold{i_fold+1}_sensitivity.csv')
        sensitivity_df = pd.DataFrame(columns=['Epoch', 'Delta_Mean', 'Delta_Min', 'Delta_Max', 'DFGU_la_1D', 'DFGU_la_2D'])
        sensitivity_df.to_csv(sensitivity_log_path, index=False)
        # ---------------------

        # 初始化早停机制
        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)
            
        best_auc = 0.0
        prev_valid_loss = float('inf') # 用于 Loss 爆炸检测

        # 初始化敏感度分析器
        sensitivity_analyzer = SensitivityAnalyzer(model, device)
        
        # 定义 Delta 更新频率 (例如每 5 个 epoch 更新一次，或者每个 epoch 更新)
        delta_update_freq = 1
        # 动量参数
        delta_momentum = 0.9

        """Start training."""
        print('Training...')
        for epoch in range(1, hp.Epoch + 1):
            if early_stopping.early_stop == True:
                break

            """train"""
            # 训练阶段
            train_losses_in_epoch = []
            model.train()
            
            # --- 外层训练循环重构：特征权重动态更新 (Delta Update with Momentum) ---
            # 借鉴 GFAN，在训练过程中定期更新 delta
            if epoch > 0 and epoch % delta_update_freq == 0:
                print(f"Epoch {epoch}: Updating feature weights (delta) using sensitivity analysis...")
                # 为了计算全局敏感度，我们可以随机采样一个 batch 或使用一部分验证集
                # 这里为了效率，使用 train_loader 的第一个 batch
                # 注意：binary_search_sensitivity 需要 eval 模式，但 update_model_delta 会处理
                try:
                    sample_data = next(iter(train_loader)).to(device)
                    # 使用当前模型的预测作为伪标签 (自监督) 或者真实标签
                    # 这里使用真实标签 cls_y，但需要注意维度匹配
                    # binary_search_sensitivity 期望 target 用于计算 loss
                    # 我们传入 criterion = Loss (CrossEntropy)
                    
                    # 临时切换到 eval 模式进行敏感度分析
                    model.eval()
                    # 计算新的敏感度
                    # 注意：binary_search_sensitivity 内部计算的是基于单个 batch 的敏感度
                    # 为了更稳定，最好是对多个 batch 取平均，这里简化为单 batch
                    # binary_search_sensitivity 默认不修改模型参数，只返回 sensitivity
                    # 我们手动实现动量更新
                    new_sensitivity = sensitivity_analyzer.binary_search_sensitivity(sample_data, sample_data.cls_y, Loss)
                    new_sensitivity = torch.clamp(new_sensitivity, 0.1, 10.0) # 截断
                    
                    # 动量更新: delta = momentum * delta + (1 - momentum) * new_sensitivity
                    model.delta.data = delta_momentum * model.delta.data + (1 - delta_momentum) * new_sensitivity
                    
                    # 记录一下 delta 的统计信息
                    current_delta_mean = model.delta.mean().item()
                    current_delta_min = model.delta.min().item()
                    current_delta_max = model.delta.max().item()
                    
                    print(f"  Delta updated (Momentum={delta_momentum}): Mean={current_delta_mean:.4f}, Min={current_delta_min:.4f}, Max={current_delta_max:.4f}")
                    
                    # 记录 DFGU 权重
                    # 获取第一层 DFGU 的权重作为代表
                    la_1D_val = torch.sigmoid(model.dfgus[0].la_1D).item()
                    la_2D_val = torch.sigmoid(model.dfgus[0].la_2D).item()
                    print(f"  DFGU Weights (Layer 0): la_1D={la_1D_val:.4f}, la_2D={la_2D_val:.4f}")
                    
                    # 写入敏感度日志
                    sensitivity_log_row = pd.DataFrame({
                        'Epoch': [epoch],
                        'Delta_Mean': [current_delta_mean],
                        'Delta_Min': [current_delta_min],
                        'Delta_Max': [current_delta_max],
                        'DFGU_la_1D': [la_1D_val],
                        'DFGU_la_2D': [la_2D_val]
                    })
                    sensitivity_log_row.to_csv(sensitivity_log_path, mode='a', header=False, index=False)
                    
                    # 恢复训练模式
                    model.train()
                except Exception as e:
                    print(f"  Warning: Failed to update delta: {e}")
                    import traceback
                    traceback.print_exc()
                    model.train()
            # ---------------------------------------------------

            for data in train_loader:
                optimizer.zero_grad()

                data = data.to(device)
                predicted_y= model(data)
                # 计算损失
                train_loss = Loss(predicted_y, data.cls_y)
                train_losses_in_epoch.append(train_loss.item())
                # 反向传播和参数更新
                train_loss.backward()
                # 梯度裁剪：防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # scheduler.step() # CyclicLR should step after batch if cycle_momentum=False/True depending on config, but standard is batch-level or epoch-level. 
                # PyTorch CyclicLR doc: "This scheduler should be stepped after each batch."
                # However, if OOM occurs, we might need to reduce batch size or optimize memory.
                # For now, let's keep it but ensure gradients are cleared.
                # optimizer.zero_grad() is called at start of loop.
                scheduler.step()
                
                # Explicitly clear graph to save memory
                del predicted_y, train_loss

            train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss

            """valid"""
            # 验证阶段
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for data in valid_loader:

                    data = data.to(device)
                    valid_scores = model(data)
                    
                    valid_labels = data.cls_y
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)

            # 计算验证集指标
            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')
            print(print_msg)

            # --- Loss 爆炸检测逻辑 ---
            if epoch > 1:
                # 检查逻辑：如果 Valid_Loss 超过上一轮的 10 倍且 Valid_AUC 跌至 0.5 (附近)
                # 考虑到 Valid_AUC 跌至 0.5 意味着模型退化为随机猜测
                # 0.55 是一个稍微宽松的阈值
                if valid_loss_a_epoch > 10 * prev_valid_loss and AUC_dev < 0.55:
                    print(f"\n{'!'*40}")
                    print(f"WARNING: Loss Explosion Detected at Epoch {epoch}!")
                    print(f"  Valid Loss: {valid_loss_a_epoch:.5f} (Previous: {prev_valid_loss:.5f})")
                    print(f"  Valid AUC: {AUC_dev:.5f}")
                    print(f"  Recording current delta state for debugging...")
                    print(f"{'!'*40}\n")
                    
                    # 记录当前的 delta 权重状态
                    explosion_log_path = os.path.join(training_logs_dir, f'{DATASET}_fold{i_fold+1}_epoch{epoch}_explosion_delta.csv')
                    pd.DataFrame(model.delta.data.cpu().numpy()).to_csv(explosion_log_path, header=False, index=False)
                    print(f"  Explosion delta saved to {explosion_log_path}")
            
            # 更新 prev_valid_loss
            prev_valid_loss = valid_loss_a_epoch
            # ------------------------

            # Append results to CSV
            # 将当前Epoch的结果写入CSV
            new_row = pd.DataFrame({
                'Epoch': [epoch],
                'Train_Loss': [train_loss_a_epoch],
                'Valid_Loss': [valid_loss_a_epoch],
                'Valid_AUC': [AUC_dev],
                'Valid_PRC': [PRC_dev],
                'Valid_Accuracy': [Accuracy_dev],
                'Valid_Precision': [Precision_dev],
                'Valid_Recall': [Reacll_dev]
            })
            new_row.to_csv(csv_file_path, mode='a', header=False, index=False)

            '''save checkpoint and make decision when early stop'''
            # 检查是否早停，并保存最佳模型
            early_stopping(Accuracy_dev, model, epoch)
            
            # --- 新增：在保存最佳模型时，同时也保存对应的 delta 向量 ---
            if AUC_dev > best_auc:
                best_auc = AUC_dev
                delta_save_path = os.path.join(training_logs_dir, f'{DATASET}_fold{i_fold+1}_best_delta.csv')
                # 保存 delta
                pd.DataFrame(model.delta.data.cpu().numpy()).to_csv(delta_save_path, header=False, index=False)
                print(f"  New best AUC: {best_auc:.5f}, Delta saved to {delta_save_path}")
            # --------------------------------------------------------

        '''load best checkpoint'''
        # 加载验证集上表现最好的模型权重
        model.load_state_dict(torch.load(early_stopping.savepath + f'/valid_best_checkpoint-{str(device).replace(":", "_")}.pth', weights_only=True))

        '''test model'''
        # 在训练集、验证集和测试集上进行最终测试
        trainset_test_stable_results, _, _, _, _, _ = test_model(
            model, train_loader, save_path, DATASET, Loss, device, dataset_class="Train", FOLD_NUM=1, MIF=True)
        validset_test_stable_results, _, _, _, _, _ = test_model(
            model, valid_loader, save_path, DATASET, Loss, device, dataset_class="Valid", FOLD_NUM=1, MIF=True)
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=1, MIF=True)
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    # 显示所有折的平均结果
    show_result(DATASET, Accuracy_List_stable, Precision_List_stable,
                Recall_List_stable, AUC_List_stable, AUPR_List_stable, Ensemble=False)
    
    # --- 新增：保存本次实验的平均结果和参数到全局日志 ---
    log_file = 'experiment_log.csv'
    # 计算平均值
    avg_acc = np.mean(Accuracy_List_stable)
    avg_auc = np.mean(AUC_List_stable)
    avg_aupr = np.mean(AUPR_List_stable)
    avg_recall = np.mean(Recall_List_stable)
    avg_precision = np.mean(Precision_List_stable)
    
    # 准备参数字典
    hp_dict = {k: v for k, v in hp.__dict__.items() if not k.startswith('__')}
    
    # 准备记录行
    record = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Dataset': DATASET,
        'Loss_Type': LOSS,
        'Avg_Accuracy': avg_acc,
        'Avg_AUC': avg_auc,
        'Avg_AUPR': avg_aupr,
        'Avg_Recall': avg_recall,
        'Avg_Precision': avg_precision,
    }
    # 合并超参数
    record.update(hp_dict)
    
    # 写入 CSV (使用 pandas)
    df = pd.DataFrame([record])
    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False)
    else:
        df.to_csv(log_file, mode='a', header=False, index=False)
    
    print(f"Experiment results and hyperparameters saved to {log_file}")
    

def ensemble_run_MIF_model(SEED, DATASET, K_Fold, device):
    """
    运行MIF-DTI集成模型
    功能：加载之前K折训练保存的K个模型，对测试集进行集成预测。
    """
    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    # 加载数据集
    assert DATASET in ["DrugBank", "BIOSNAP", "Davis"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    '''shuffle data'''
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    '''split dataset to train&validation set and test set'''
    # 划分测试集（使用最后20%的数据）
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:-1]
    print('Number of Test set: {}'.format(len(test_data_list)))

    save_path = f"./{DATASET}/ensemble"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    '''Data Preparation'''
    # 准备蛋白质和配体数据
    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein.pkl'
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi.pkl'
    if os.path.exists(ligand_path):
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    torch.cuda.empty_cache()  
    
    # 构建测试集Dataset和DataLoader
    test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=device)
    
    # test_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
    #                                  collate_fn=collate_fn, drop_last=True)
    test_dataset_loader = pyg_loader.DataLoader(test_dataset, batch_size=1,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)

    # 加载所有K折模型
    model = []
    for i in range(K_Fold):
        model.append(MIFDTI().to(device))
        '''MIF-DTI K-Fold train process is necessary'''
        try:
            model[i].load_state_dict(torch.load(
                f'./{DATASET}/{i+1}' + f'/valid_best_checkpoint-{device}.pth', map_location=torch.device(device)))   #加载对应权重
        except FileNotFoundError as e:
            print('-'* 25 + 'ERROR' + '-'*25)
            error_msg = 'Load pretrained model error: \n' + \
                        str(e) + \
                        '\n' + 'MIFDTI K-Fold train process is necessary'
            print(error_msg)
            print('-'* 55)
            exit(1)

    Loss = PolyLoss(weight_loss=weight_loss,
                    DEVICE=device, epsilon=hp.loss_epsilon)

#   testdataset_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
#       model, test_dataset_loader, save_path, DATASET, Loss, device, dataset_class="Test", save=True, FOLD_NUM=K_Fold)
    
    # 测试集成模型
    testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=K_Fold, MIF=True)
    
    # 显示结果
    show_result(DATASET, Accuracy_test, Precision_test,
                Recall_test, AUC_test, PRC_test, Ensemble=True)
    