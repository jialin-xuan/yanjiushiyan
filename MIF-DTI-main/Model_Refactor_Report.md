
# 模型改造汇总与分析报告

## 1. 项目改造概览

本项目旨在对原有的 **MIF-DTI** (Multi-Information Fusion Drug-Target Interaction) 模型进行深度改造，以增强其解释性（Interpretability）和鲁棒性。我们借鉴了 **GFAN** (Graph Feature Attention Network) 的思想，引入了探测式推理（Probe-based Inference）机制，实现了特征权重的动态更新和敏感度分析。

改造的主要目标包括：
1.  **底层组件升级**：引入门控机制（Gated-CNN）和动态融合单元（DFGU）。
2.  **核心解释器算法**：实现 $O(\log F)$ 的二分查找敏感度扫描。
3.  **闭环探测系统**：构建支持自我解释和动态调整的推理流程。
4.  **外层训练循环**：实现特征权重的周期性更新（Delta Update）。

---

## 2. 核心模块改造详解

### 2.1 底层组件升级 (`layers.py`)

**改造内容**：
*   **GatedConv1d (门控一维卷积)**：替换了原有的简单 `nn.Conv1d`。
    *   **原理**：包含两个并行的卷积分支，一个提取特征（线性），一个生成门控权重（Sigmoid）。输出为 `Linear * Sigmoid(Gate)`。
    *   **优势**：增强了序列特征提取的非线性能力和选择性。
*   **DFGU (动态门控融合单元)**：新增模块。
    *   **原理**：通过可学习的标量参数 `la_1D` 和 `la_2D` 对 1D 序列特征和 2D 图特征进行加权融合。
    *   **自适应机制**：接收 `sensitivity_2D` 参数，当图特征敏感度较低时，自动抑制 2D 路径权重，增强 1D 路径权重，提高模型在图结构信息缺失时的鲁棒性。

**代码位置**：
*   [layers.py](layers.py) : `class GatedConv1d`, `class DFGU`

### 2.2 模型架构重构 (`model.py`)

**改造内容**：
*   **特征掩码 (Delta Mask)**：
    *   在 `MIFDTI` 类中引入了 `self.delta` 参数（初始全 1）。
    *   在 `forward` 函数中，对所有输入特征（原子特征、氨基酸特征、序列嵌入）应用 `self.delta` 进行加权，实现特征筛选。
*   **DFGU 集成**：
    *   在每一层 `MIFBlock` 后，使用 `DFGU` 融合序列分支和图分支的特征。
    *   在最终输出层，使用 `DFGU` 替代原有的 `RESCAL` 或 `PoolAttention`，融合药物和蛋白质的最终表示。
*   **探测式推理 (Probe Inference)**：
    *   新增 `probe_inference` 方法。
    *   **流程**：快速前向传播 -> 二分查找敏感度扫描 -> 更新 Delta -> 二次前向传播（自适应融合）-> 输出预测 + 解释。

**代码位置**：
*   [model.py](model.py) : `class MIFDTI`, `MIFDTI.forward`, `MIFDTI.probe_inference`

### 2.3 核心解释器算法 (`utils/sensitivity_analyzer.py`)

**改造内容**：
*   **SensitivityAnalyzer 类**：新增工具类。
*   **二分查找敏感度扫描 (Binary Search Sensitivity Scan)**：
    *   **原理**：递归地将特征空间二分屏蔽，观察预测误差的变化。
    *   **优化**：将原本 $O(F)$ 的逐一剔除法优化为 $O(\log F)$，大幅提升了解释器在复杂数据集上的运行效率。
*   **Delta Update**：
    *   提供了 `update_model_delta` 方法，用于在训练过程中更新模型的特征权重。

**代码位置**：
*   [utils/sensitivity_analyzer.py](utils/sensitivity_analyzer.py) : `SensitivityAnalyzer.binary_search_sensitivity`

### 2.4 外层训练循环重构 (`RunModel.py`)

**改造内容**：
*   **Wrapper 机制**：
    *   在标准训练循环（Epoch Loop）之外，增加了特征权重更新逻辑。
*   **Delta Update Loop**：
    *   在每个 Epoch 开始时，调用 `SensitivityAnalyzer` 计算全局特征敏感度，并更新模型的 `self.delta`。
    *   确保模型在训练过程中能够动态抑制冗余特征，关注关键子结构。

**代码位置**：
*   [RunModel.py](RunModel.py) : `run_MIF_model` 函数中的训练循环部分

---

## 3. 算法原理分析

### 3.1 为什么选择 $O(\log F)$ 二分扫描？
传统的特征重要性分析通常采用逐一剔除法（Leave-One-Out），即每次屏蔽一个特征，计算误差变化。对于高维特征（如药物分子指纹、蛋白质序列嵌入），特征维度 $F$ 可能达到数千甚至数万，导致计算成本极高。
二分查找扫描利用了特征的稀疏性假设（即只有少数特征对结果有显著影响）。通过递归地屏蔽特征块，如果某一块屏蔽后误差无显著变化，则直接跳过该块，从而将搜索复杂度降低到对数级别。

### 3.2 自适应多模态融合 (Adaptive Multi-modal Fusion)
MIF-DTI 模型同时利用了 1D 序列信息（SMILES, Amino Acid Sequence）和 2D 图结构信息（Molecular Graph, Protein Graph）。在某些情况下（如构象不确定、图构建噪声大），2D 信息可能不可靠。
通过引入 `DFGU` 和敏感度反馈机制，模型可以动态感知 2D 特征的可靠性（`sensitivity_2D`）。当 2D 特征敏感度低时，模型自动降低其权重，更多地依赖 1D 序列特征，从而提高了模型在各种数据质量下的鲁棒性。

---

## 4. 总结

本次改造成功将 MIF-DTI 升级为一个**具备自我解释能力的闭环系统**。模型不仅能够进行准确的 DTI 预测，还能通过探测式推理提供预测依据（特征敏感度），并在训练过程中自我优化特征权重。这一架构为后续的药物发现提供了更透明、更可靠的 AI 工具。
