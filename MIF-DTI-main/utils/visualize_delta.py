
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import torch
import joblib
import sys

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MIFDTI
from utils.DataPrepare import get_kfold_data, shuffle_dataset
from utils import protein_init, ligand_init, ProteinMoleculeDataset
import torch_geometric.loader as pyg_loader

# 假设 config 中 hidden_channels = 200 (默认)
# 如果不是，需要从 hyperparameters.json 读取
HIDDEN_CHANNELS = 200 

def visualize_delta(dataset_name, fold, top_k=20):
    """
    绘制 Delta 热图并找出权重最高的前 K 个特征索引
    """
    log_dir = 'results/training_logs/'
    
    # 1. 加载训练过程中的敏感度日志 (用于分析变化趋势)
    sensitivity_log_path = os.path.join(log_dir, f'{dataset_name}_fold{fold}_sensitivity.csv')
    if os.path.exists(sensitivity_log_path):
        print(f"Loading sensitivity log from {sensitivity_log_path}")
        df_log = pd.read_csv(sensitivity_log_path)
        
        # 绘制 Delta 统计值变化趋势
        plt.figure(figsize=(12, 6))
        plt.plot(df_log['Epoch'], df_log['Delta_Mean'], label='Mean')
        plt.plot(df_log['Epoch'], df_log['Delta_Min'], label='Min')
        plt.plot(df_log['Epoch'], df_log['Delta_Max'], label='Max')
        plt.xlabel('Epoch')
        plt.ylabel('Delta Value')
        plt.title(f'Feature Sensitivity (Delta) Statistics over Epochs ({dataset_name} Fold {fold})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, f'{dataset_name}_fold{fold}_delta_stats.png'))
        print(f"Saved delta statistics plot to {log_dir}")
        plt.close()

        # 绘制 DFGU 权重变化趋势
        plt.figure(figsize=(12, 6))
        plt.plot(df_log['Epoch'], df_log['DFGU_la_1D'], label='la_1D (Sequence)')
        plt.plot(df_log['Epoch'], df_log['DFGU_la_2D'], label='la_2D (Graph)')
        plt.xlabel('Epoch')
        plt.ylabel('DFGU Weight (Sigmoid)')
        plt.title(f'DFGU Weights Adaptation over Epochs ({dataset_name} Fold {fold})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, f'{dataset_name}_fold{fold}_dfgu_weights.png'))
        print(f"Saved DFGU weights plot to {log_dir}")
        plt.close()
    
    # 2. 加载最佳模型的 Delta 向量 (用于热图和 Top-K)
    best_delta_path = os.path.join(log_dir, f'{dataset_name}_fold{fold}_best_delta.csv')
    if os.path.exists(best_delta_path):
        print(f"Loading best delta from {best_delta_path}")
        # 读取 CSV，注意之前保存时没有 header 和 index
        delta_df = pd.read_csv(best_delta_path, header=None)
        delta_vec = delta_df.values.flatten()
        
        # 找出权重最高的前 K 个特征
        top_indices = np.argsort(delta_vec)[-top_k:][::-1]
        top_values = delta_vec[top_indices]
        
        print(f"\nTop {top_k} Most Sensitive Features:")
        for idx, val in zip(top_indices, top_values):
            print(f"Feature Index: {idx}, Sensitivity (Delta): {val:.4f}")
            
        # 绘制 Delta 向量分布直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(delta_vec, bins=50, kde=True)
        plt.title(f'Distribution of Feature Sensitivities (Delta) at Best AUC ({dataset_name} Fold {fold})')
        plt.xlabel('Sensitivity Value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(log_dir, f'{dataset_name}_fold{fold}_delta_dist.png'))
        print(f"Saved delta distribution plot to {log_dir}")
        plt.close()
        
        # 绘制 Delta 热图 (Reshape 为矩阵以便观察)
        dim = len(delta_vec)
        cols = 20
        rows = dim // cols
        if rows * cols == dim:
            delta_matrix = delta_vec.reshape(rows, cols)
            plt.figure(figsize=(12, 8))
            sns.heatmap(delta_matrix, cmap='viridis', annot=False)
            plt.title(f'Feature Sensitivity Heatmap ({dataset_name} Fold {fold})')
            plt.savefig(os.path.join(log_dir, f'{dataset_name}_fold{fold}_delta_heatmap.png'))
            print(f"Saved delta heatmap to {log_dir}")
            plt.close()
        else:
            print(f"Cannot reshape delta vector of size {dim} into nice matrix (cols={cols}). Skipping heatmap.")
            
    else:
        print(f"Best delta file not found: {best_delta_path}")

def visualize_best_sample_path(dataset_name, fold, device='cpu'):
    """
    针对测试集预测最准的样本，可视化其二分查找路径锁定的特征块
    """
    print("\nVisualizing Binary Search Path for Best Sample...")
    log_dir = 'results/training_logs/'
    model_path = f'./{dataset_name}/{fold}/valid_best_checkpoint-{str(device).replace(":", "_")}.pth'
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        return

    # 1. 加载数据
    print("Loading data...")
    dir_input = ('./DataSets/{}.txt'.format(dataset_name))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    
    # 简单的预处理和加载 (简化版，仅用于演示)
    # 注意：为了准确找到测试集，需要复用 RunModel 中的 split 逻辑
    # 这里直接取最后一部分作为测试集
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:-1]
    
    protein_path = f'./DataSets/Preprocessed/{dataset_name}-protein-new.pkl'
    protein_dict = joblib.load(protein_path)
    ligand_path = f'./DataSets/Preprocessed/{dataset_name}-ligand-hi-new.pkl'
    ligand_dict = joblib.load(ligand_path)
    
    test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=device)
    test_loader = pyg_loader.DataLoader(test_dataset, batch_size=1, shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])
    
    # 2. 加载模型
    print("Loading model...")
    model = MIFDTI(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. 寻找预测最准的样本 (预测概率与真实标签差异最小)
    best_sample_idx = -1
    min_diff = 1.0
    best_data = None
    best_pred = -1
    best_label = -1
    
    print("Finding best sample...")
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            prob = torch.softmax(output, dim=1)[:, 1].item() # 正类概率
            label = data.cls_y.item()
            
            diff = abs(prob - label)
            if diff < min_diff:
                min_diff = diff
                best_sample_idx = i
                best_data = data
                best_pred = prob
                best_label = label
                
        # 限制搜索范围，避免太慢
        if i > 100: 
            break
            
    print(f"Best Sample Found: Index {best_sample_idx}, Label: {best_label}, Pred: {best_pred:.4f}, Diff: {min_diff:.4f}")
    
    # 4. 可视化二分查找路径
    # 这里我们模拟二分查找过程，记录每一步的误差变化，并可视化
    # 复用 SensitivityAnalyzer 的逻辑，但增加 logging
    from utils.sensitivity_analyzer import SensitivityAnalyzer
    analyzer = SensitivityAnalyzer(model, device)
    
    # 定义 target (使用真实标签)
    target = torch.tensor([best_label], device=device, dtype=torch.long)
    # 对于 CrossEntropyLoss，target 需要是 LongTensor 且形状为 (N,)
    # 但 binary_search_sensitivity 内部使用的是 model(data) 和 target 计算 loss
    # 如果 Loss 是 CELoss，它期望 output 是 logits, target 是 labels
    from LossFunction import CELoss
    criterion = CELoss(weight_CE=None, DEVICE=device)
    
    # 计算敏感度并记录路径
    # 为了可视化路径，我们需要修改 binary_search_sensitivity 或在这里重新实现带 log 的版本
    # 这里我们在 analyzer 外部手动进行一次带记录的扫描
    
    original_error = analyzer._get_prediction_error(best_data, target, criterion)
    num_features = model.delta.shape[0]
    path_log = []
    
    def recursive_scan_log(start, end, current_base_error, depth=0):
        if start >= end:
            return

        # 记录当前区间和误差
        path_log.append({
            'Depth': depth,
            'Start': start,
            'End': end,
            'Base_Error': current_base_error
        })
        
        # 模拟判断逻辑 (这里简化，只记录访问过的节点)
        # 实际的二分查找会根据 region_importance 决定是否深入
        # 我们这里只记录前几层，或者根据之前的 delta 结果来决定“感兴趣”的区域
        
        if end - start <= num_features // 8: # 限制深度，避免日志过多
            return

        mid = (start + end) // 2
        recursive_scan_log(start, mid, current_base_error, depth+1)
        recursive_scan_log(mid, end, current_base_error, depth+1)

    recursive_scan_log(0, num_features, original_error)
    
    # 绘制路径图
    # 我们可以绘制一个树状图或区间图，显示哪些区域被访问了
    # 这里简单绘制一个区间覆盖图
    plt.figure(figsize=(12, 6))
    for entry in path_log:
        depth = entry['Depth']
        start = entry['Start']
        end = entry['End']
        plt.hlines(y=-depth, xmin=start, xmax=end, linewidth=2, color='skyblue')
        plt.vlines(x=start, ymin=-depth-0.2, ymax=-depth+0.2, color='gray', linestyle=':')
        plt.vlines(x=end, ymin=-depth-0.2, ymax=-depth+0.2, color='gray', linestyle=':')
        plt.text((start+end)/2, -depth+0.1, f"[{start},{end})", ha='center', fontsize=8)
        
    plt.title(f'Binary Search Scan Path for Best Sample (Index {best_sample_idx})')
    plt.xlabel('Feature Index')
    plt.ylabel('Recursion Depth (Negative)')
    plt.yticks([])
    plt.grid(True, axis='x')
    plt.savefig(os.path.join(log_dir, f'{dataset_name}_fold{fold}_best_sample_path.png'))
    print(f"Saved binary search path plot to {log_dir}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Delta and Sensitivity Logs')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., Davis)')
    parser.add_argument('fold', type=int, help='Fold number (e.g., 1)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    visualize_delta(args.dataset, args.fold)
    visualize_best_sample_path(args.dataset, args.fold, device)
