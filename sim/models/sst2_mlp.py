'''Multi-Layer Perceptron for SST-2 Text Classification
适用于 SST-2 数据集（二分类情感分析数据集）的 MLP 模型

模型结构：
- Embedding层：将词索引转换为词向量
- 平均池化：将变长序列转换为固定长度向量
- 全连接层：进行分类

Model           Param
MLP             ~(vocab_size * embed_dim + hidden_dim * embed_dim + num_classes * hidden_dim)
'''
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, num_classes=2, vocab_size=10000, embed_dim=128, hidden_dim=64, max_seq_len=128):
        """
        Args:
            num_classes (int): 分类数量，SST-2为2（正/负情感）
            vocab_size (int): 词汇表大小，默认10000
            embed_dim (int): 词向量维度，默认128
            hidden_dim (int): 隐藏层维度，默认64
            max_seq_len (int): 最大序列长度，默认128
        """
        super(MLP, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 全连接层
        self.layer_input = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer_hidden = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len)，包含词索引
            lengths: 每个序列的实际长度（可选），用于掩码填充位置
        Returns:
            logits: 分类logits，形状为 (batch_size, num_classes)
        """
        # 词嵌入: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        x = self.embedding(x)
        
        # 平均池化：对序列维度求平均，忽略填充位置
        # (batch_size, seq_len, embed_dim) -> (batch_size, embed_dim)
        if lengths is not None:
            # 创建掩码，忽略填充位置
            mask = (torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
            mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            x = x * mask  # 将填充位置置零
            x = x.sum(dim=1) / lengths.unsqueeze(-1).float()  # 按实际长度求平均
        else:
            # 如果没有提供lengths，直接对所有位置求平均
            x = x.mean(dim=1)
        
        # 全连接层
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        
        return x


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    
    # 测试模型
    vocab_size = 10000
    batch_size = 32
    seq_len = 50
    
    model = MLP(num_classes=2, vocab_size=vocab_size)
    count_parameters(model)
    
    # 创建测试输入
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.randint(10, seq_len, (batch_size,))
    
    print("\n测试前向传播（带lengths）:")
    output = model(x, lengths)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    print("\n测试前向传播（不带lengths）:")
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    try:
        print(summary(model, x, lengths, show_input=True))
    except:
        print("\n注意: 无法显示模型摘要，可能需要安装 pytorch_model_summary")

