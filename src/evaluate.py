import torch
from config import *
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_topk
from tokenizer import JiebaTokenizer
from tqdm import tqdm

def evaluate(model, dataloader, device):
    top1_acc_count, top5_acc_count = 0, 0
    total_count = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluating: '):
            inputs, targets = inputs.to(device), targets.to(device)
            top5_indices_list = predict_topk(model, inputs)
            for target, top5_indices in zip(targets, top5_indices_list):
                total_count += 1
                if target == top5_indices[0]:
                    top1_acc_count += 1
                if target in top5_indices:
                    top5_acc_count += 1
    top1_acc = top1_acc_count / total_count
    top5_acc = top5_acc_count / total_count
    return top1_acc, top5_acc

def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)
    print('vocabulary load success!')

    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))
    print('model load success!')

    test_dataloader = get_dataloader(train=False)
    top1_acc, top5_acc = evaluate(model, test_dataloader, device)

    print('evaluate result:')
    print('top1_acc: ', top1_acc)
    print('top5_acc: ', top5_acc)

if __name__ == '__main__':
    run_evaluate()
