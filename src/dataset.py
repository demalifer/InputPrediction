import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from watchdog.utils import BaseThread

from config import *

class InputMethodDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input = torch.tensor(self.data[index]['input'], dtype=torch.long)
        target = torch.tensor(self.data[index]['target'], dtype=torch.long)
        return input, target

def get_dataloader(train=True):
    path = PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = InputMethodDataset(path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)

    for input, target in train_dataloader:
        print(input.shape, target.shape)
        break