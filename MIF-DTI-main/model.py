# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from layers import *
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool
                                )
from config import hyperparameter

class MIF_conv_block(nn.Module):
    """
    MIF卷积块：包含GATConv层，LayerNorm和SAGPooling
    功能：对输入图特征进行卷积、归一化和池化，提取全局图嵌入。
    """
    def __init__(self, in_channels=200, out_channels=200, num_heads=4, dropout=0.3):
        super(MIF_conv_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        # 图注意力卷积层 (GAT)
        # 作用：通过注意力机制聚合邻居节点特征
        self.conv = GATConv(self.in_channels, self.out_channels//self.num_heads, self.num_heads, dropout=self.dropout)
        # 层归一化
        self.norm = LayerNorm(self.in_channels)
        # 自注意力图池化 (SAGPooling)
        # 作用：基于节点重要性评分进行池化，减少图的规模
        self.readout = SAGPooling(self.out_channels, min_score=-1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # 归一化后经过ELU激活函数
        x = F.elu(self.norm(x, batch))
        # 经过GAT卷积，聚合邻居特征
        x = self.conv(x, edge_index, edge_attr)
        # 池化操作，获取更新后的节点特征x，以及池化后的批次信息batch
        # readout返回多个值，这里我们主要关注x和batch，以及用于全局池化的x_batch
        x, _, _, x_batch, _, _ = self.readout(x, edge_index, edge_attr=edge_attr, batch=batch)
        # 全局加和池化，得到整个图的特征向量
        global_graph_emb = global_add_pool(x, x_batch)
        return x, global_graph_emb


class MIFBlock(nn.Module):
    """
    MIF核心模块：处理药物和蛋白质的相互作用
    功能：分别处理药物和蛋白质的图结构数据，并利用注意力机制交互两者的信息。
    """
    def __init__(self, in_channels=200, out_channels=200, num_heads=5, dropout=0.4):
        super(MIFBlock, self).__init__()
        
        self.hidden_channels = out_channels // (num_heads*2)
        # 药物内部卷积：处理药物分子图内部的相互作用
        self.drug_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.1)
        # 蛋白质内部卷积：处理蛋白质结构图内部的相互作用
        self.prot_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.3)
        # 交互卷积（药物与蛋白质之间）：处理药物原子和蛋白质氨基酸之间的相互作用
        # 输入为二分图结构 (source, target)，这里用于跨模态信息传递
        self.inter_conv = GATConv((in_channels, in_channels), self.hidden_channels, num_heads, dropout=dropout)
        # 层归一化
        self.drug_norm = LayerNorm(out_channels)
        self.prot_norm = LayerNorm(out_channels)
        # 池化层：进一步提取特征并降低维度
        self.drug_pool = GATConv(out_channels, out_channels//num_heads, num_heads)
        self.prot_pool = SAGPooling(out_channels, min_score=-1)
        # self.prot_pool = GATConv(out_channels, out_channels//num_heads, num_heads)

    def forward(self, atom_x, atom_edge_index, bond_x, atom_batch, \
                aa_x, aa_edge_index, aa_edge_attr, aa_batch, m2p_edge_index):
        
        # 保存残差连接的输入
        atom_x_res = atom_x
        aa_x_res = aa_x

        # 1. 药物特征更新
        # 药物内部特征提取 (Intra-molecular interaction)
        atom_intra_x = self.drug_conv(atom_x, atom_edge_index, bond_x)
        # 药物-蛋白质交互特征提取 (Inter-molecular interaction)
        # m2p_edge_index[[1,0]] 表示从蛋白质流向药物的信息
        atom_inter_x = self.inter_conv((aa_x, atom_x), m2p_edge_index[[1,0]])
        # 拼接内部和交互特征
        atom_x_tmp = torch.cat([atom_intra_x, atom_inter_x], -1)
        # 归一化和激活
        atom_x = F.elu(self.drug_norm(atom_x_tmp, atom_batch))

        # 2. 蛋白质特征更新
        # 蛋白质内部特征提取 (Intra-molecular interaction)
        aa_intra_x = self.prot_conv(aa_x, aa_edge_index, aa_edge_attr)
        # 蛋白质-药物交互特征提取 (Inter-molecular interaction)
        # m2p_edge_index 表示从药物流向蛋白质的信息
        aa_inter_x = self.inter_conv((atom_x, aa_x), m2p_edge_index)
        # 拼接内部和交互特征
        aa_x_tmp = torch.cat([aa_intra_x, aa_inter_x], -1)
        # 归一化和激活
        aa_x = F.elu(self.prot_norm(aa_x_tmp, aa_batch))

        # 3. 池化操作
        # 药物特征进一步处理
        atom_x = self.drug_pool(atom_x, atom_edge_index, bond_x)
        # 蛋白质特征池化
        aa_x, _, _, aa_batch, _, _ = self.prot_pool(aa_x, aa_edge_index, edge_attr=aa_edge_attr, batch=aa_batch)
        
        # 4. 残差连接和Dropout
        # 将原始输入加回到处理后的特征上，防止梯度消失
        atom_x = F.dropout(atom_x_res+F.elu(atom_x), 0.1, self.training)
        aa_x = F.dropout(aa_x_res+F.elu(aa_x), 0.1, self.training)
        
        # 全局池化，获取图级别的表示
        drug_global_repr = global_add_pool(atom_x, atom_batch)
        prot_global_repr = global_add_pool(aa_x, aa_batch)

        return atom_x, aa_x, drug_global_repr, prot_global_repr

class MIFBlock_1D(nn.Module):
    """
    1D MIF模块：处理序列数据（如药物SMILES和蛋白质序列）
    功能：利用CNN提取局部序列特征，利用Multi-Head Attention进行特征融合。
    """
    def __init__(self, input_dim=200, conv=50, drug_kernel=[4, 6, 8], prot_kernel=[4, 8, 12]):
        super(MIFBlock_1D, self).__init__()
        self.attention_dim = conv * 4
        self.mix_attention_head = 5

        # CNN层，用于提取局部特征
        # 不同大小的卷积核用于提取不同尺度的特征
        self.Drug_CNNs = get_CNNs(input_dim, conv, drug_kernel)
        self.Protein_CNNs = get_CNNs(input_dim, conv, prot_kernel)

        # 多头注意力层，用于特征融合
        # 实现药物和蛋白质序列特征的交叉注意力机制
        self.mix_attention_layer = nn.MultiheadAttention(self.attention_dim, self.mix_attention_head, batch_first=True, dropout=0.3)

    def forward(self, drugembed, proteinembed):

        # 调整维度以适应CNN: [batch_size, seq_len, embed_dim] -> [batch_size, embed_dim, seq_len] 
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # 经过CNN提取特征
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # 调整维度以适应Attention: [batch_size, embed_dim, seq_len] -> [batch_size, seq_len, embed_dim]
        drugConv = drugConv.permute(0, 2, 1)
        proteinConv = proteinConv.permute(0, 2, 1)

        # 交叉注意力 (Cross Attention)
        # drug_att: 药物关注蛋白质特征
        drug_att, _ = self.mix_attention_layer(drugConv, proteinConv, proteinConv)
        # protein_att: 蛋白质关注药物特征
        protein_att, _ = self.mix_attention_layer(proteinConv, drugConv, drugConv)

        # 残差连接：结合原始CNN特征和注意力特征
        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        # 最大池化：提取最显著的特征
        drugPool, _ = torch.max(drugConv, dim=1)
        proteinPool, _ = torch.max(proteinConv, dim=1)

        return drugConv, proteinConv, drugPool, proteinPool


from utils.sensitivity_analyzer import SensitivityAnalyzer

class MIFDTI(nn.Module):
    """
    MIF-DTI主模型
    功能：融合图结构特征和序列特征，进行药物-靶标相互作用预测。
    """
    def __init__(self, depth=3, device='cuda:0'):
        super(MIFDTI, self).__init__()

        self.drug_in_channels = 43
        self.prot_in_channels = 33
        self.prot_evo_in_channels = 1280
        self.hidden_channels = 200
        self.depth = depth
        self.device = device
        
        # 探测模式标志
        self.probe_mode = False

        # 分子特征编码器
        # 原子类型嵌入
        self.atom_type_encoder = Embedding(20, self.hidden_channels)
        # 原子特征MLP
        self.atom_feat_encoder = MLP([self.drug_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 
        # 化学键嵌入
        self.bond_encoder = Embedding(10, self.hidden_channels)

        # 蛋白质特征编码器
        # 进化信息MLP (Evolutionary features)
        self.prot_evo = MLP([self.prot_evo_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 
        # 氨基酸特征MLP (Amino acid features)
        self.prot_aa = MLP([self.prot_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 

        # 编码器模块列表（图神经网络部分）
        # 堆叠多层MIFBlock，用于深度特征提取
        self.blocks = nn.ModuleList([MIFBlock() for _ in range(depth)])

        # 序列嵌入
        self.drug_seq_emb = nn.Embedding(65, self.hidden_channels, padding_idx=0)
        self.prot_seq_emb = nn.Embedding(26, self.hidden_channels, padding_idx=0)
        # 编码器模块列表（序列部分）
        # 堆叠多层MIFBlock_1D，处理序列信息
        self.blocks_1D = nn.ModuleList([MIFBlock_1D() for _ in range(depth)])
        
        # 动态门控融合单元列表
        self.dfgus = nn.ModuleList([DFGU() for _ in range(depth)])
        
        # 特征掩码 delta
        self.delta = nn.Parameter(torch.ones(self.hidden_channels))

        # 最终的预测层（RESCAL）
        # 使用RESCAL张量分解方法计算药物和蛋白质表示的交互得分
        # 在探测式推理中，我们可能需要使用 DFGU 替代 RESCAL，或者在 RESCAL 之前加入 DFGU 融合
        # 根据指令，改造模型解码器（将 RESCAL 或 PoolAttention 替换为 DFGU）
        # 这里我们用 DFGU 融合药物和蛋白质的最终表示，然后通过 MLP 输出概率
        self.final_fusion = DFGU()
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_channels // 2, 2)
        )
        
        # 保留 RESCAL 作为备选或对比，或者如果指令明确替换，则移除
        # self.attn = RESCAL(self.hidden_channels, self.depth)
        
        self.to(device)
        
        # 初始化敏感度分析器
        self.sensitivity_analyzer = SensitivityAnalyzer(self, device)

    def probe_inference(self, data):
        """
        探测式推理流程 (Probe-based Inference)
        步骤 1：进行初次快速前向传播
        步骤 2：调用 binary_search_sensitivity 计算当前样本的 1D 和 2D 模态敏感度
        步骤 3：将敏感度转化为动态权重 la_1D 和 la_2D (DFGU内部逻辑)
        步骤 4：将权重传入 DFGU 进行加权融合，输出最终 DTI 概率
        """
        self.probe_mode = True
        
        # 步骤 1 & 2: 计算敏感度
        # 注意：binary_search_sensitivity 内部会多次调用 forward
        # 为了避免无限递归，我们需要在 forward 中处理 probe_mode
        # 或者我们在这里手动控制 delta 更新
        
        # 假设我们关注的是最终分类结果的置信度作为 target 代理
        # 或者我们可以构造一个伪 target
        # 为了简化，我们使用模型自身的预测作为 target (自监督方式探测稳定性)
        with torch.no_grad():
            initial_output = self.forward(data)
            # 假设输出是 logits，转换为概率
            initial_probs = torch.softmax(initial_output, dim=1)
            # 使用预测概率最大的类别作为伪标签
            pseudo_target = initial_probs.clone() 
        
        # 定义 criterion，例如 KL 散度或 MSE，用于衡量屏蔽特征后输出的变化
        criterion = nn.MSELoss()
        
        # 计算敏感度
        sensitivity = self.sensitivity_analyzer.binary_search_sensitivity(data, pseudo_target, criterion)
        
        # 步骤 3: 转化为动态权重 (在 DFGU 中自动处理，这里我们需要确保 delta 被更新)
        # binary_search_sensitivity 默认不修改模型参数，只返回 sensitivity
        # 我们需要临时更新 self.delta 以影响 forward 中的 DFGU
        original_delta = self.delta.data.clone()
        self.delta.data = sensitivity
        
        # 步骤 4: 再次前向传播，输出最终 DTI 概率和解释
        final_output = self.forward(data)
        
        # 恢复原始 delta
        self.delta.data = original_delta
        self.probe_mode = False
        
        return final_output, sensitivity

    def forward(self,data):

        # 获取分子数据
        atom_x, atom_x_feat, smiles_x, atom_edge_index, bond_x, mol_node_levels = \
            data.mol_x, data.mol_x_feat, data.mol_smiles_x, data.mol_edge_index, data.mol_edge_attr, data.mol_node_levels
        # 获取蛋白质数据 (氨基酸)
        aa_x, aa_evo_x, seq_x, aa_edge_index, aa_edge_weight = \
            data.prot_node_aa, data.prot_node_evo, data.prot_seq_x, data.prot_edge_index, data.prot_edge_weight, \
        # 获取Batch信息
        atom_batch, aa_batch = data.mol_x_batch, data.prot_node_aa_batch
        # 双向图边索引
        m2p_edge_index = data.m2p_edge_index

        # ----------------------------------------------------------------------------------------------------------------
        # 敏感度自适应 (Adaptive Sensitivity Handling)
        # 如果模型有 pre-calculated sensitivity info，可以传递给 DFGU
        # 这里为了演示，我们假设 sensitivity 可以从 delta 中推断 (delta 值越小说明越不重要，反之亦然，或者反过来)
        # 根据 GFAN 定义：delta = E_k / E_0. E_k 是 mask 后的 error。
        # 如果 mask 后 error 变大 (E_k > E_0)，说明特征重要，delta > 1。
        # 如果 mask 后 error 不变 (E_k = E_0)，说明特征不重要，delta = 1。
        # 如果 mask 后 error 变小 (E_k < E_0)，说明特征是有害噪音，delta < 1。
        
        # 计算当前图特征的平均敏感度 (简化逻辑：取 delta 的均值作为整体图特征可靠性指标)
        sensitivity_2D = torch.mean(self.delta)
        # ----------------------------------------------------------------------------------------------------------------

        # 分子特征初始化：结合原子类型嵌入和原子特征
        atom_x = self.atom_type_encoder(atom_x.squeeze()) + self.atom_feat_encoder(atom_x_feat)
        bond_x = self.bond_encoder(bond_x)
        
        # 应用特征掩码 delta
        atom_x = atom_x * self.delta
                
        # 蛋白质特征初始化：结合氨基酸特征和进化信息
        aa_x = self.prot_aa(aa_x) + self.prot_evo(aa_evo_x)
        # 应用特征掩码 delta
        aa_x = aa_x * self.delta

        # 使用RBF核处理边权重
        aa_edge_attr = rbf(aa_edge_weight, D_max=1.0, D_count=self.hidden_channels, device=self.device)

        # 编码过程 (Encoding)
        drug_repr_graph = []
        prot_repr_graph = []
        
        # 1. 图神经网络部分 (Graph Branch)
        for i in range(self.depth):
            # 每一层MIFBlock都会更新节点特征并提取全局表示
            out = self.blocks[i](atom_x, atom_edge_index, bond_x, atom_batch, \
                                 aa_x, aa_edge_index, aa_edge_attr, aa_batch, \
                                 m2p_edge_index)
            atom_x, aa_x, drug_global_repr, prot_global_repr = out
            # 这里似乎只取特定层级的原子特征作为全局表示的一部分（可能是为了多尺度融合）
            drug_global_repr = atom_x[mol_node_levels==2]
            drug_repr_graph.append(drug_global_repr)
            prot_repr_graph.append(prot_global_repr)

        # 2. 序列部分 (Sequence Branch)
        atom_x_seq = self.drug_seq_emb(smiles_x)
        aa_x_seq = self.prot_seq_emb(seq_x)
        
        # 应用特征掩码 delta 到序列特征
        atom_x_seq = atom_x_seq * self.delta
        aa_x_seq = aa_x_seq * self.delta

        drug_repr_seq = []
        prot_repr_seq = []
        for i in range(self.depth):
            # 每一层MIFBlock_1D都会更新序列特征并提取池化后的表示
            out_seq = self.blocks_1D[i](atom_x_seq, aa_x_seq)
            atom_x_seq, aa_x_seq, drug_seq_pool, prot_seq_pool = out_seq
            drug_repr_seq.append(drug_seq_pool)
            prot_repr_seq.append(prot_seq_pool)

        # 使用 DFGU 融合图特征和序列特征
        drug_repr = []
        prot_repr = []
        for i in range(self.depth):
            # 传入 sensitivity_2D 以启用自适应切换
            d_fused = self.dfgus[i](drug_repr_seq[i], drug_repr_graph[i], sensitivity_2D=sensitivity_2D)
            p_fused = self.dfgus[i](prot_repr_seq[i], prot_repr_graph[i], sensitivity_2D=sensitivity_2D)
            drug_repr.append(d_fused)
            prot_repr.append(p_fused)

        # 堆叠所有深度和所有分支的表示
        # 维度变换: [batch_size, num_layers, hidden_channels]
        drug_repr = torch.stack(drug_repr, dim=-2)
        prot_repr = torch.stack(prot_repr, dim=-2)

        # 联合注意力机制 (Co-attn) 计算最终得分
        # 替换原有的 RESCAL 或 PoolAttention，使用 DFGU 进行最终融合
        
        # 简化处理：将所有深度的特征聚合（例如求和或取最后一层）
        drug_final = torch.mean(drug_repr, dim=1) # [batch_size, hidden_channels]
        prot_final = torch.mean(prot_repr, dim=1) # [batch_size, hidden_channels]
        
        # 使用 DFGU 融合药物和蛋白质的表示
        # 这里 DFGU 接受 1D (药物) 和 2D (蛋白质) 特征
        # 但实际上 DFGU 设计是融合 1D 序列和 2D 图，这里我们将 drug 和 prot 视为两个模态
        # 为了复用 DFGU，我们将 drug 视为 1D 特征，prot 视为 2D 特征（或者反之，视具体语义而定）
        # 假设 drug 是我们要重点分析的小分子 (1D SMILES)，prot 是复杂的结构 (2D Graph)
        fused_repr = self.final_fusion(drug_final, prot_final, sensitivity_2D=sensitivity_2D)
        
        # 通过分类器输出概率
        scores = self.classifier(fused_repr)

        return scores

def get_m2p_edge_from_batch(atom_batch, aa_batch, node_level=None):

    mask = atom_batch.unsqueeze(1) == aa_batch.unsqueeze(0)  # (num_a_nodes, num_b_nodes) 的bool矩阵
    if node_level is not None:
        mask = mask * (node_level==1).unsqueeze(1)
    a_idx, b_idx = torch.nonzero(mask, as_tuple=True)
    edge_list = torch.stack([a_idx, b_idx], dim=0)
    return edge_list
