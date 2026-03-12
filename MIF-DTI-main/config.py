# -*- coding:utf-8 -*-


class hyperparameter():
    """
    超参数配置类
    功能：存储模型训练和测试所需的各种超参数。
    """
    def __init__(self):
        self.Learning_rate = 1e-4  # 学习率：控制参数更新的步长
        self.Epoch = 200           # 训练轮数：总共训练多少个epoch
        self.Batch_size = 64      # 批次大小：每次迭代训练的样本数量
        # self.Batch_size = 64      # 批次大小
        self.Patience = 50         # 早停耐心值：验证集性能多少轮不提升则停止训练 原50
        self.decay_interval = 10   # 学习率衰减间隔（目前代码中似乎使用CyclicLR，此参数可能未被使用）
        self.lr_decay = 0.5        # 学习率衰减率（同上）
        self.weight_decay = 1e-4   # 权重衰减：L2正则化系数，防止过拟合
        self.embed_dim = 64        # 嵌入维度：基础特征向量的维度
        self.protein_kernel = [4, 8, 12] # 蛋白质CNN卷积核大小列表：用于提取不同尺度的局部特征
        self.drug_kernel = [4, 6, 8]     # 药物CNN卷积核大小列表
        self.conv = 50             # 卷积通道数：CNN层的输出通道数
        self.char_dim = 64         # 字符嵌入维度（可能用于序列编码）
        self.loss_epsilon = 1      # PolyLoss的epsilon参数：调节多项式项的权重