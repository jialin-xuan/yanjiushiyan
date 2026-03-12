# MIF-DTI 项目全流程运行文档

本文档详细描述了经过模型改造后的 MIF-DTI 项目的完整运行流程，包括数据流向、核心处理逻辑、输入输出及新增的闭环反馈机制。

---

## 1. 系统架构概览

改造后的系统是一个具备**自我解释**和**动态优化**能力的闭环系统。核心差异在于引入了**特征敏感度分析 (Sensitivity Analysis)** 和 **动态门控融合 (DFGU)**，使得模型在训练过程中能够根据特征重要性自动调整权重。

### 核心数据流图

```mermaid
graph TD
    A[原始数据 (SMILES, Sequence)] --> B[数据预处理 (Graph & Seq Construction)]
    B --> C[DataLoader (Batching)]
    C --> D{MIF-DTI 模型}
    
    subgraph Model_Internal [模型内部处理]
        D1[特征编码 (Atom, Bond, AA)] --> D2[特征掩码 (Delta Mask)]
        D2 --> D3_1[图分支 (MIFBlock)]
        D2 --> D3_2[序列分支 (MIFBlock_1D)]
        D3_1 & D3_2 --> D4[动态融合 (DFGU)]
        D4 --> D5[最终分类器 (MLP)]
    end
    
    D --> E[预测输出 (Logits)]
    E --> F[损失计算 (Loss)]
    
    subgraph Feedback_Loop [闭环反馈机制]
        F --> G[敏感度分析 (Sensitivity Analyzer)]
        G --> H[二分查找扫描 (Binary Search)]
        H --> I[动量更新 (Momentum Update)]
        I --> D2
    end
```

---

## 2. 详细流程说明

### 2.1 数据输入与预处理 (Input & Preprocessing)

*   **数据源**: `DataSets/{DATASET}.txt`
    *   **格式**: `SMILES字符串 蛋白质序列 亲和力标签(0/1)`
*   **预处理 (`DataPrepare.py` & `utils/`)**:
    *   **分子处理**:
        *   **1D**: SMILES 序列化 (用于 `MIFBlock_1D`)。
        *   **2D**: 使用 BRICS 算法构建分子图 (Clique Graph)，提取原子特征和化学键信息 (用于 `MIFBlock`)。
    *   **蛋白质处理**:
        *   **1D**: 氨基酸序列化。
        *   **2D**: 基于序列构建蛋白质接触图 (Contact Map)，提取氨基酸物理化学性质和进化信息。
*   **输出**: `PyG Data` 对象，包含 `mol_x` (原子), `mol_edge_index` (键), `prot_node_aa` (氨基酸), `prot_edge_index` (残基接触) 等。

### 2.2 模型前向传播 (Forward Pass)

数据进入 `MIFDTI` 模型 (`model.py`) 后，经历以下阶段：

1.  **特征初始化与掩码**:
    *   输入特征经过 `Embedding` 或 `MLP` 映射到 `hidden_channels` 维度。
    *   **关键操作**: 应用全局特征掩码 `self.delta`。
        *   公式: $X_{masked} = X_{raw} \times \delta$
        *   作用: 根据训练反馈抑制噪声特征，增强关键子结构。

2.  **双分支深度特征提取**:
    *   模型堆叠了 `depth` (默认3) 层模块。
    *   **图分支 (Graph Branch)**: 使用 GAT (Graph Attention Network) 和 Co-Attention 处理药物-靶标图结构交互。
    *   **序列分支 (Sequence Branch)**: 使用 CNN 提取局部模式，Self-Attention 提取长程依赖。

3.  **动态门控融合 (DFGU)**:
    *   每一层输出的 1D 和 2D 特征通过 `DFGU` 进行融合。
    *   **自适应机制**: 接收 `sensitivity_2D` 参数 (由 `delta` 均值推导)。
        *   若图特征敏感度低 (不可靠)，自动降低 2D 权重，通过 `sigmoid` 门控机制平滑切换到 1D 主导。

4.  **最终预测**:
    *   融合后的药物特征和蛋白质特征再次通过 `DFGU` 结合。
    *   通过 `MLP` 分类器输出最终的预测分数 (Logits)。

### 2.3 训练与反馈循环 (Training & Feedback Loop)

在 `RunModel.py` 的训练循环中，新增了**特征权重动态更新**逻辑：

1.  **标准训练**:
    *   每个 Batch 进行前向传播 -> 计算 Loss (CrossEntropy) -> 反向传播 -> 优化器更新模型参数。

2.  **Delta 更新 (Meta-Learning Step)**:
    *   **触发时机**: 每个 Epoch 开始时 (或每 N 个 Epoch)。
    *   **采样**: 从训练集中抽取一个 Batch。
    *   **敏感度计算**:
        *   调用 `SensitivityAnalyzer.binary_search_sensitivity`。
        *   使用 **$O(\log F)$ 二分查找算法**，递归地屏蔽特征区域，观察 Loss 变化。
        *   输出特征敏感度向量 $S$。
    *   **动量更新 (Momentum Update)**:
        *   调用 `update_model_delta(alpha=0.9)`。
        *   公式: $\delta_{new} = 0.9 \cdot \delta_{old} + 0.1 \cdot S$
        *   **数值保护**: 结果被截断在 `[0.1, 10.0]` 之间，防止权重爆炸或消失。
    *   **效果**: `self.delta` 被更新，直接影响下一个 Epoch 的特征筛选。

### 2.4 探测式推理 (Probe Inference)

在推理或验证阶段，模型可以开启**探测模式** (`model.probe_inference`)：

1.  **初次预测**: 快速前向传播，获取初始预测结果作为伪标签。
2.  **即时解释**: 针对当前样本运行二分查找敏感度扫描。
3.  **动态调整**: 根据当前样本的敏感度，临时调整 `delta` 和 `DFGU` 权重。
4.  **最终输出**: 再次前向传播，输出经过自适应调整后的高置信度预测，并返回特征重要性热图。

---

## 3. 输入输出总结

| 阶段 | 输入 (Input) | 输出 (Output) | 备注 |
| :--- | :--- | :--- | :--- |
| **预处理** | 原始文本 (SMILES, Seq) | PyG Data Objects (Graphs) | 包含原子/氨基酸特征矩阵 |
| **训练** | Batch Data | 1. Model Weights (包含 Delta)<br>2. Training Logs (Loss, AUC)<br>3. Sensitivity Logs (Delta Mean/Min/Max) | 实时监控特征演化 |
| **Delta更新**| Sample Batch | Updated `self.delta` Parameter | 这里的输出直接修改模型状态 |
| **推理** | Single/Batch Data | 1. Prediction Score (Prob)<br>2. Sensitivity Map (Interpretation) | 可视化用 |

## 4. 关键文件索引

*   **流程控制**: [RunModel.py](RunModel.py) (训练循环与 Delta 更新调度)
*   **模型架构**: [model.py](model.py) (MIFDTI, Probe Inference)
*   **基础组件**: [layers.py](layers.py) (DFGU, GatedConv1d)
*   **解释器**: [utils/sensitivity_analyzer.py](utils/sensitivity_analyzer.py) (Binary Search, Momentum Update)
*   **数据准备**: [DataPrepare.py](DataPrepare.py) & [utils/DataSetsFunction.py](utils/DataSetsFunction.py)

---

这份文档涵盖了模型改造后的完整生命周期，从数据加载到闭环训练，再到可解释性推理，清晰地展示了数据如何在各个模块间流转并被处理。
