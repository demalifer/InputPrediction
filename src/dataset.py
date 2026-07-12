import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from watchdog.utils import BaseThread

from config import *

# 本文件负责把 preprocess.py 生成的 jsonl 样本包装成 PyTorch Dataset/DataLoader。
# 每一行样本的格式是：
# {"input": [token_id_1, ..., token_id_5], "target": next_token_id}

class InputMethodDataset(Dataset):
    def __init__(self, path):
        # pandas 按 jsonl 逐行读取，每一行都是一个训练样本。
        # 转成 list[dict] 后，__getitem__ 可以按下标直接取样本。
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        # DataLoader 需要通过 __len__ 知道整个数据集有多少条样本。
        return len(self.data)

    def __getitem__(self, index):
        # input 是长度为 SEQ_LEN 的 token id 序列，target 是下一个 token 的 id。
        # dtype=torch.long 是 nn.Embedding 和 CrossEntropyLoss 对类别 id 的要求。
        input = torch.tensor(self.data[index]['input'], dtype=torch.long)
        target = torch.tensor(self.data[index]['target'], dtype=torch.long)
        return input, target

def get_dataloader(train=True):
    # train=True 读取训练集，否则读取测试集。
    path = PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = InputMethodDataset(path)
    # shuffle=True 会打乱样本顺序，训练时有利于优化；测试时虽然不必要，但不影响准确率统计。
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

if __name__ == '__main__':
    # 单独运行本文件时，简单检查 DataLoader 输出的 batch 形状是否符合预期。
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)

    for input, target in train_dataloader:
        print(input.shape, target.shape)
        break
