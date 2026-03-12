
import torch
import torch.nn as nn
import numpy as np
import copy
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    """
    敏感度分析器
    功能：通过特征掩码分析模型输入的特征重要性。
    支持：二分查找扫描 (O(log F)) 优化，替代逐一剔除法。
    """
    def __init__(self, model, device='cuda:0'):
        self.model = model
        self.device = device
        self.model.eval()

    def _get_prediction_error(self, data, target, criterion=None):
        """
        计算预测误差
        """
        with torch.no_grad():
            output = self.model(data)
            # 假设是分类任务，默认使用 BCEWithLogitsLoss 或类似的逻辑
            # 这里为了通用性，如果未提供 criterion，则使用预测概率的负对数似然作为“误差”代理
            # 或者简单地返回 Loss
            if criterion:
                loss = criterion(output, target)
                return loss.item()
            else:
                # 默认行为：假设 output 是 logits，计算 sigmoid 后的熵或者直接用 logits 的 magnitude
                # 为了简化，这里假设 target 是 label，计算 BCE
                # 注意：需要确保 data 和 target 的维度匹配
                # 如果没有 target，我们可以计算输出的置信度变化，但这通常需要 target
                # 暂时假设调用者会提供 criterion 和 target
                return 0.0

    def calculate_sensitivity_linear(self, data, target, criterion):
        """
        线性扫描 (O(F))：逐一屏蔽特征计算敏感度 (Legacy, for reference)
        """
        original_error = self._get_prediction_error(data, target, criterion)
        num_features = self.model.hidden_channels # 假设我们要分析的是 hidden_channels 维度的 delta
        
        sensitivity = torch.ones(num_features).to(self.device)
        
        # 保存原始 delta
        original_delta = self.model.delta.data.clone()
        
        for k in range(num_features):
            # 屏蔽第 k 个特征
            mask = torch.ones(num_features).to(self.device)
            mask[k] = 0
            self.model.delta.data = original_delta * mask
            
            new_error = self._get_prediction_error(data, target, criterion)
            
            # 敏感度 = 屏蔽后的误差 / 原始误差
            # 如果屏蔽后误差变大，说明该特征重要，比值 > 1
            if original_error != 0:
                sensitivity[k] = new_error / original_error
            else:
                sensitivity[k] = 1.0
                
        # 恢复原始 delta
        self.model.delta.data = original_delta
        return sensitivity

    def binary_search_sensitivity(self, data, target, criterion, threshold=0.01):
        """
        二分查找敏感度扫描 (O(log F))
        原理：递归地将特征空间二分，如果屏蔽某一半特征导致的误差变化不显著，则认为该区域特征不重要，停止深入扫描。
        
        Args:
            data: 模型输入数据
            target: 目标标签
            criterion: 损失函数
            threshold: 误差变化阈值，低于此阈值认为不重要
            
        Returns:
            sensitivity: 特征敏感度得分向量
        """
        original_error = self._get_prediction_error(data, target, criterion)
        if original_error == 0:
            logger.warning("Original error is 0, cannot calculate sensitivity ratio.")
            return torch.ones_like(self.model.delta)

        num_features = self.model.delta.shape[0]
        sensitivity = torch.ones(num_features).to(self.device)
        
        # 保存原始 delta
        original_delta_val = self.model.delta.data.clone()

        def scan_range(start, end, current_base_error):
            """
            递归扫描范围 [start, end)
            """
            if start >= end:
                return

            # 如果只有一个特征，直接计算
            if end - start == 1:
                # 屏蔽这一个特征
                mask = torch.ones(num_features).to(self.device)
                mask[start] = 0
                self.model.delta.data = original_delta_val * mask
                
                new_error = self._get_prediction_error(data, target, criterion)
                ratio = new_error / original_error
                sensitivity[start] = ratio
                return

            # 尝试屏蔽整个区间 [start, end)
            mask = torch.ones(num_features).to(self.device)
            mask[start:end] = 0
            self.model.delta.data = original_delta_val * mask
            
            region_error = self._get_prediction_error(data, target, criterion)
            
            # 如果屏蔽整个区域后的误差变化与基准误差相比变化不大，说明这个区域整体不重要
            # 注意：这里的逻辑需要精细调整。通常二分查找用于定位“哪里重要”。
            # 如果屏蔽整个区域导致的误差飙升（region_error >> original_error），说明这里面有重要特征，需要继续细分。
            # 如果 region_error ≈ original_error，说明屏蔽这些特征没影响，它们不重要，可以停止递归。
            
            # 计算区域重要性得分
            region_importance = region_error / original_error
            
            # 如果区域重要性显著（例如大于 1 + threshold），则深入
            # 或者如果我们需要精细的每个特征的得分，我们可能还是需要细分，但可以利用 heuristic 剪枝
            # 为了实现真正的 O(log F) 效果，我们假设特征是稀疏的，只有少数特征重要。
            
            if abs(region_importance - 1.0) < threshold:
                # 该区域不重要，该区域内所有特征敏感度设为 1.0 (基准)
                sensitivity[start:end] = 1.0
            else:
                # 该区域重要，继续二分
                mid = (start + end) // 2
                scan_range(start, mid, original_error)
                scan_range(mid, end, original_error)

        # 开始递归扫描
        # 注意：为了更精确，我们可以先全量扫描一遍判断是否需要深入，或者直接开始
        # 这里为了演示 O(log F) 的潜力，我们采用“如果不重要就跳过”的策略
        scan_range(0, num_features, original_error)

        # 恢复原始 delta
        self.model.delta.data = original_delta_val
        
        return sensitivity

    def update_model_delta(self, data, target, criterion, alpha=0.9):
        """
        计算敏感度并更新模型的 delta 参数
        引入动量更新机制，提高特征权重演化的平滑度。
        """
        # binary_search_sensitivity 默认不修改 delta
        sensitivity = self.binary_search_sensitivity(data, target, criterion)
        
        # 动量更新 (Momentum Update)
        # delta_new = alpha * delta_old + (1 - alpha) * sensitivity
        old_delta = self.model.delta.data
        new_delta = alpha * old_delta + (1 - alpha) * sensitivity
        
        # 数值保护：确保更新后的 delta 始终处于安全数值区间
        new_delta = torch.clamp(new_delta, 0.1, 10.0)
        
        # 更新 self.delta
        self.model.delta.data = new_delta
        
        # 日志记录：监控特征极化进度
        logger.info(f"Model delta updated (alpha={alpha}). "
                    f"Mean: {new_delta.mean().item():.4f}, "
                    f"Min: {new_delta.min().item():.4f}, "
                    f"Max: {new_delta.max().item():.4f}")
        return new_delta
