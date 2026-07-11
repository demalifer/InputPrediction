import torch
import torch.nn as nn

from config import *

class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        super(InputMethodModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=EMBEDDING_SIZE)
        self.rnn = nn.RNN(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=vocab_size)

    def forward(self, input):
        embed = self.embedding(input)
        output, _ = self.rnn(embed)
        result = self.linear(output[:,-1,:])
        return result

if __name__ == '__main__':
    vocab_size = 1000
    input = torch.randint(1000, size=(64, 5))
    model = InputMethodModel(vocab_size)
    output = model(input)
    print(output.size())
