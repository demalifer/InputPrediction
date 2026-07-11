import torch
import torch.nn as nn

class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        super(InputMethodModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=128)
        self.rnn = nn.RNN(input_size=128, hidden_size=256, batch_first=True)
        self.linear = nn.Linear(in_features=256, out_features=vocab_size)

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
