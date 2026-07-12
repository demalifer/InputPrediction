import torch
from config import *
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_topk
from tokenizer import JiebaTokenizer
from tqdm import tqdm

# 本文件负责加载训练好的模型，在测试集上计算 top-1 和 top-5 准确率。
# top-1：模型概率最高的候选正好是答案。
# top-5：正确答案出现在概率最高的 5 个候选里。

def evaluate(model, dataloader, device):
    # 分别累计 top-1 命中数、top-5 命中数，以及总样本数。
    top1_acc_count, top5_acc_count = 0, 0
    total_count = 0

    # eval() 会关闭 dropout/batchnorm 等训练行为；本模型没有这些层，但评估时仍应设置。
    model.eval()
    # no_grad() 禁用梯度记录，评估更省显存和时间。
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluating: '):
            inputs, targets = inputs.to(device), targets.to(device)
            # predict_topk 返回每条样本最可能的 k 个 token id。
            top5_indices_list = predict_topk(model, inputs)
            for target, top5_indices in zip(targets, top5_indices_list):
                total_count += 1
                # 第一个候选就是 top-1 预测结果。
                if target == top5_indices[0]:
                    top1_acc_count += 1
                # 只要正确答案在前 5 个候选里，就算 top-5 命中。
                if target in top5_indices:
                    top5_acc_count += 1
    top1_acc = top1_acc_count / total_count
    top5_acc = top5_acc_count / total_count
    return top1_acc, top5_acc

def run_evaluate():
    # 有 GPU 就用 CUDA，否则回退到 CPU。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 词表大小决定模型输出层维度，所以评估时必须加载训练时同一份词表。
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)
    print('vocabulary load success!')

    # 构建与训练阶段相同结构的模型，然后加载 best_model.pt 中保存的参数。
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))
    print('model load success!')

    # 测试集来自 preprocess.py 生成的 data/processed/test.jsonl。
    test_dataloader = get_dataloader(train=False)
    top1_acc, top5_acc = evaluate(model, test_dataloader, device)

    print('evaluate result:')
    print('top1_acc: ', top1_acc)
    print('top5_acc: ', top5_acc)

if __name__ == '__main__':
    # 命令行直接执行 python src/evaluate.py 时进入完整评估流程。
    run_evaluate()
