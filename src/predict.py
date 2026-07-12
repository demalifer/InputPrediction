import torch
import jieba
from config import *
from model import InputMethodModel

def predict_topk(model, input, k=5):
    model.eval()
    with torch.no_grad():
        output = model(input)
    top_indices = torch.topk(output, k=k).indices
    return top_indices.tolist()

def predict(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(MODEL_DIR/VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab_list = [token.strip() for token in f.readlines()]
    id2word = {id: word for id, word in enumerate(vocab_list)}
    word2id = {word: id for id, word in enumerate(vocab_list)}
    print('vocabulary load success!')

    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))
    print('model load success!')

    tokens = jieba.lcut(text)
    ids = [word2id.get(token, word2id.get(UNK_TOKEN)) for token in tokens]
    input = torch.tensor([ids], dtype=torch.long).to(device)

    top_indices_list = predict_topk(model, input)
    top_tokens = [id2word[id] for id in top_indices_list[0]]

    return top_tokens


if __name__ == '__main__':
    text = '我们'
    top5_tokens = predict(text)
    print(top5_tokens)