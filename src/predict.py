import torch
import jieba
from config import *
from model import InputMethodModel
from tokenizer import JiebaTokenizer

def predict_topk(model, input, k=5):
    model.eval()
    with torch.no_grad():
        output = model(input)
    top_indices = torch.topk(output, k=k).indices
    return top_indices.tolist()

def predict(text, model, tokenizer, k, device):

    ids = tokenizer.encode(text)

    input = torch.tensor([ids], dtype=torch.long).to(device)

    top_indices_list = predict_topk(model, input, k=k)
    top_tokens = [tokenizer.id2word[id] for id in top_indices_list[0]]

    return top_tokens

def run_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)
    print('vocabulary load success!')

    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
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
        top_tokens = predict(input_history, model, tokenizer, 5, device)
        print('top tokens:', top_tokens)

if __name__ == '__main__':
    run_predict()