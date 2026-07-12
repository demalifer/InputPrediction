import torch
from torch import nn, optim
from tqdm import tqdm

from config import *
from dataset import get_dataloader
from model import InputMethodModel

from torch.utils.tensorboard import SummaryWriter
import time

def train_one_epoch(model, train_loader, loss , optimizer, device):
    model.train()

    total_loss = 0
    for input, target in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        input, target = input.to(device), target.to(device)
        output = model(input)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss_value.item()

    return total_loss / len(train_loader)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_dataloader(train=True)
    with open(MODEL_DIR/VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab_list = [token.strip() for token in f.readlines()]
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    min_loss = float('inf')
    for epoch in range(EPOCHS):
        print('='*15, f'EPOCH {epoch+1}', '='*15)
        this_loss = train_one_epoch(model, train_loader, loss, optimizer, device)
        print('the loss of this epoch is : ', this_loss)

        writer.add_scalar('loss', this_loss, epoch + 1)

        if this_loss < min_loss:
            min_loss = this_loss
            torch.save(model.state_dict(), MODEL_DIR / BEST_MODEL)
            print('The best model has been saved!')
        else:
            print('This model is not the best, and it has not been saved!')
    writer.close()

if __name__ == '__main__':
    train()