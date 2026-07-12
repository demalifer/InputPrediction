import torch
from torch import nn, optim
from tqdm import tqdm

from config import *
from dataset import get_dataloader
from model import InputMethodModel

from torch.utils.tensorboard import SummaryWriter
import time
from tokenizer import JiebaTokenizer

# 本文件负责训练 RNN 输入法预测模型，并按训练损失保存当前最优权重到 models/best_model.pt。

def train_one_epoch(model, train_loader, loss , optimizer, device):
    # train() 会开启训练模式；如果模型包含 dropout/batchnorm，这一步尤其重要。
    model.train()

    total_loss = 0
    for input, target in tqdm(train_loader, desc='Training'):
        # 清空上一轮 batch 累积的梯度。
        optimizer.zero_grad()
        # 把数据搬到与模型相同的设备上。
        input, target = input.to(device), target.to(device)
        # 前向传播得到每个候选 token 的 logits。
        output = model(input)
        # CrossEntropyLoss 接收 logits 和类别 id，会内部做 log-softmax。
        loss_value = loss(output, target)
        # 反向传播计算梯度，然后 optimizer.step() 更新参数。
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        # item() 把单个 tensor 标量取成 Python 数字，便于累计日志。
        total_loss += loss_value.item()

    # 返回本 epoch 的平均 batch loss。
    return total_loss / len(train_loader)

def train():
    # 有 GPU 就用 CUDA，否则使用 CPU。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 训练数据来自 data/processed/train.jsonl。
    train_loader = get_dataloader(train=True)

    # 加载词表以确定模型输出维度，也保证训练和推理 token id 一致。
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)

    # 初始化模型、损失函数和 Adam 优化器。
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # TensorBoard 日志按当前时间新建目录，避免覆盖历史训练记录。
    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    # 用训练损失作为保存最佳模型的标准：loss 更低就覆盖 best_model.pt。
    min_loss = float('inf')
    for epoch in range(EPOCHS):
        print('='*15, f'EPOCH {epoch+1}', '='*15)
        this_loss = train_one_epoch(model, train_loader, loss, optimizer, device)
        print('the loss of this epoch is : ', this_loss)

        # 记录每一轮的训练损失，之后可通过 TensorBoard 查看曲线。
        writer.add_scalar('loss', this_loss, epoch + 1)

        if this_loss < min_loss:
            min_loss = this_loss
            torch.save(model.state_dict(), MODEL_DIR / BEST_MODEL)
            print('The best model has been saved!')
        else:
            print('This model is not the best, and it has not been saved!')
    writer.close()

if __name__ == '__main__':
    # 命令行直接执行 python src/train.py 时启动训练流程。
    train()
