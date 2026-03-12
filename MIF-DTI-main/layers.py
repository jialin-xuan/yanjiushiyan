import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv1d(nn.Module):
    """
    门控一维卷积 (Gated CNN)
    功能：包含两个并行的Conv1d层，一个用于提取特征，一个用于生成门控权重。
    输出 = 线性分支 * sigmoid(门控分支)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(GatedConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.gate = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        return self.conv1d(x) * torch.sigmoid(self.gate(x))



class DFGU(nn.Module):
    """
    动态门控融合单元 (Dynamic Fusion Gated Unit)
    功能：接收1D序列特征和2D图特征，根据动态标量系数进行加权融合。
    支持：自适应切换机制 (Adaptive Switching Mechanism)，根据特征敏感度自动调整权重。
    """
    def __init__(self):
        super(DFGU, self).__init__()
        # 动态标量系数
        self.la_1D = nn.Parameter(torch.tensor(0.5))
        self.la_2D = nn.Parameter(torch.tensor(0.5))
        
        # 自适应阈值，用于判断敏感度是否过低
        self.sensitivity_threshold = 0.2

    def forward(self, feat_1D, feat_2D, sensitivity_2D=None):
        """
        Args:
            feat_1D: 1D序列特征
            feat_2D: 2D图特征
            sensitivity_2D: (Optional) 2D特征的平均敏感度得分
        """
        # 如果提供了敏感度信息，且2D特征敏感度过低（意味着图结构信息不可靠或缺失）
        # 则自动降低2D路径的权重，增强1D路径的权重
        if sensitivity_2D is not None and sensitivity_2D < self.sensitivity_threshold:
             # 动态调整权重：抑制 la_2D，增强 la_1D
             # 注意：为了保持梯度的流动，这里使用软调整而非硬截断
             weight_2D = torch.sigmoid(self.la_2D) * sensitivity_2D # 降低
             weight_1D = torch.sigmoid(self.la_1D) + (1.0 - sensitivity_2D) # 增强
             
             return weight_1D * feat_1D + weight_2D * feat_2D
        else:
             # 默认行为
             return self.la_1D * feat_1D + self.la_2D * feat_2D



class MLP(nn.Module):
    """
    多层感知机 (Multi-Layer Perceptron)
    功能：构建全连接层网络，支持LayerNorm和ReLU激活。
    """
    def __init__(self, dims, out_norm=False, in_norm=False, bias=True): #L=nb_hidden_layers
        super().__init__()
        # 构建线性层列表
        list_FC_layers = [ nn.Linear(dims[idx-1], dims[idx], bias=bias) for idx in range(1,len(dims)) ]
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        # 输出层归一化
        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        # 输入层归一化
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        # 初始化参数
        for idx in range(self.hidden_layers+1):
            self.FC_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x):
        y = x
        # 输入层归一化
        if self.in_norm:
            y = self.in_ln(y)

        # 隐藏层前向传播：Linear -> ReLU
        for idx in range(self.hidden_layers):
            y = self.FC_layers[idx](y)
            y = F.relu(y)
        # 最后一层只有Linear
        y = self.FC_layers[-1](y)

        # 输出层归一化
        if self.out_norm:
            y = self.out_ln(y)

        return y


class CoAttentionLayer(nn.Module):
    """
    协同注意力层 (Co-Attention Layer)
    功能：计算两个特征序列之间的注意力矩阵。
    """
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        # 定义查询、键、偏置和注意力权重的参数
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        # Xavier初始化
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        # 计算Keys和Queries
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q

        # 计算激活值：Q + K + Bias
        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        # 计算注意力得分：tanh(activations) * a
        e_scores = torch.tanh(e_activations) @ self.a
        attentions = e_scores
        return attentions
    

class RESCAL(nn.Module):
    """
    RESCAL模型
    功能：基于张量分解的关系学习方法，用于计算药物和蛋白质之间的交互得分。
    """
    def __init__(self, n_features, depth):
        super().__init__()
        self.n_features = n_features
        # 协同注意力层
        self.co_attn = CoAttentionLayer(n_features)
        # 最终预测的MLP
        self.mlp = nn.Sequential(
            nn.Linear(depth*depth, 2)
        )

    def forward(self, heads, tails):
        # 计算协同注意力分数
        alpha_scores = self.co_attn(heads, tails)
        # 归一化特征
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        # 计算特征交互得分
        scores = (heads @ tails.transpose(-2, -1))
        # 结合注意力分数
        scores *= alpha_scores
        # 通过MLP进行最终分类
        scores = self.mlp(scores.reshape(scores.shape[0], -1))
        return scores
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"
    
class PoolAttention(nn.Module):
    """利用Attention进行多模态融合, `with-attn`变体的关键组件"""

    def __init__(self, n_features, num_neads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_features, num_heads=num_neads, batch_first=True)
        self.drug_norm = nn.LayerNorm(n_features)
        self.prot_norm = nn.LayerNorm(n_features)
        self.mlp = nn.Sequential(
            nn.Linear(n_features*2, n_features*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_features*2, n_features*1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_features*1, 2),
        )

    def forward(self, drug, prot):
        # 归一化
        drug = self.drug_norm(drug)
        prot = self.prot_norm(prot)
        # 自注意力机制
        drug_attn = self.attn(drug, prot, prot)[0]
        prot_attn = self.attn(prot, drug, drug)[0]
        # 池化操作：最大池化
        drug_pool = torch.max((drug+drug_attn)/2, dim=1)[0]
        prot_pool = torch.max((prot+prot_attn)/2, dim=1)[0]
        # 拼接并预测
        scores = self.mlp(torch.cat([drug_pool, prot_pool], dim=-1))
        return scores

class AttentionLayer(nn.Module):
    """
    普通注意力层
    功能：利用MultiHeadAttention融合药物和蛋白质特征。
    """
    def __init__(self, n_features, heads=4):
        super().__init__()
        self.n_features = n_features
        self.heads = heads
        self.attn = nn.MultiheadAttention(self.n_features, self.heads, batch_first=True, dropout=0.3)
        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.n_features*2, self.n_features),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.n_features, self.n_features),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.n_features, 2)
        )

    def forward(self, drug_repr, prot_repr):
        # 交叉注意力
        drug_output, _ = self.attn(drug_repr, prot_repr, prot_repr)
        prot_output, _ = self.attn(prot_repr, drug_repr, drug_repr)

        # 残差连接
        drug_output = drug_output * 0.5 + drug_repr * 0.5
        prot_output = prot_output * 0.5 + prot_repr * 0.5

        # 最大池化
        drug_pool, _ = torch.max(drug_output, dim=1)
        prot_pool, _ = torch.max(prot_output, dim=1)
        # 拼接特征
        concat_repr = torch.cat([drug_pool, prot_pool], -1)
        # 最终预测
        result = self.mlp(concat_repr)
        return result
    

def rbf(D, D_min=0., D_max=1., D_count=16, device='cpu'):
    '''
    径向基函数 (Radial Basis Function) 编码
    来源: https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    # 截断距离
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device) )
    # 生成RBF中心
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    # 计算sigma
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    # 计算高斯核
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def get_CNNs(input_dim, conv_dim, kernel):
    """
    构建CNN序列模块
    功能：构建多层门控一维卷积网络，用于提取序列特征。
    """
    return nn.Sequential(
            GatedConv1d(in_channels=input_dim, out_channels=conv_dim, kernel_size=kernel[0]),
            nn.ReLU(),
            GatedConv1d(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=kernel[1]),
            nn.ReLU(),
            GatedConv1d(in_channels=conv_dim*2, out_channels=conv_dim*4, kernel_size=kernel[2]),
            nn.ReLU(),
        )