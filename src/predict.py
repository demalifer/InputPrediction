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

def predict(text, model, id2word, word2id, k, device):

    tokens = jieba.lcut(text)
    ids = [word2id.get(token, word2id.get(UNK_TOKEN)) for token in tokens]
    input = torch.tensor([ids], dtype=torch.long).to(device)

    top_indices_list = predict_topk(model, input, k=k)
    top_tokens = [id2word[id] for id in top_indices_list[0]]

    return top_tokens

def run_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(MODEL_DIR/VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab_list = [token.strip() for token in f.readlines()]
    id2word = {id: word for id, word in enumerate(vocab_list)}
    word2id = {word: id for id, word in enumerate(vocab_list)}
    print('vocabulary load success!')

    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))
    print('model load success!')

    print('Welcome to INTELEGER input method! print q or quit to exit...')
    input_history = ''
    while True:
        user_input = input('> ')
        if user_input.strip() in ['q', 'quit']:
            print('bye!')
            break
        elif user_input.strip() == '':
            print('please input valid content!')
            continue

        input_history += user_input
        top_tokens = predict(input_history, model, id2word, word2id, 5, device)
        print('top tokens:', top_tokens)

if __name__ == '__main__':
    run_predict()