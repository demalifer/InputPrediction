import torch
import torch.nn as nn

from config import *

# 本文件定义输入法预测模型。
# 模型输入是一批 token id 序列，形状通常是 [batch_size, SEQ_LEN]。
# 模型输出是每条序列对应的下一个 token 分类分数，形状是 [batch_size, vocab_size]。

class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        super(InputMethodModel, self).__init__()
        # Embedding 把离散 token id 映射成连续向量，便于神经网络学习语义/上下文特征。
        self.embedding = nn.Embedding(vocab_size, embedding_dim=EMBEDDING_SIZE)
        # RNN 按顺序读取 token 向量；batch_first=True 表示输入形状为 [batch, seq, feature]。
        self.rnn = nn.RNN(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        # 线性层把最后一个时间步的隐藏状态映射成词表大小的分类 logits。
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=vocab_size)

    def forward(self, input):
        # input: [batch_size, seq_len]，每个元素是 token id。
        embed = self.embedding(input)
        # embed: [batch_size, seq_len, EMBEDDING_SIZE]。
        # output 保存每个时间步的隐藏状态，_ 是最后隐状态，这里不单独使用。
        output, _ = self.rnn(embed)
        # 只取最后一个时间步的输出，因为任务是根据完整上下文预测下一个 token。
        result = self.linear(output[:,-1,:])
        return result

if __name__ == '__main__':
    # 单独运行本文件时，用随机输入检查前向传播输出维度。
    vocab_size = 1000
    input = torch.randint(1000, size=(64, 5))
    model = InputMethodModel(vocab_size)
    output = model(input)
    print(output.size())
