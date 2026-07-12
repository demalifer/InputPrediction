import torch
import jieba
from config import *
from model import InputMethodModel
from tokenizer import JiebaTokenizer

# 本文件负责推理预测：给定用户已经输入的文本，返回模型认为最可能出现的后续 token。

def predict_topk(model, input, k=5):
    # 切到评估模式，并关闭梯度，推理阶段不需要反向传播。
    model.eval()
    with torch.no_grad():
        # output 是未归一化的分类分数 logits，形状为 [batch_size, vocab_size]。
        output = model(input)
    # torch.topk 直接在词表维度上取分数最高的 k 个 token id。
    top_indices = torch.topk(output, k=k).indices
    # 转成 Python list，便于后续查词表或做准确率统计。
    return top_indices.tolist()

def predict(text, model, tokenizer, k, device):

    # 先用训练阶段同一套 tokenizer 把中文文本切词并转成 token id。
    ids = tokenizer.encode(text)

    # 模型按 batch 处理输入，所以这里额外套一层列表，形成 [1, seq_len]。
    input = torch.tensor([ids], dtype=torch.long).to(device)

    # top_indices_list[0] 对应当前这一条输入文本的前 k 个预测结果。
    top_indices_list = predict_topk(model, input, k=k)
    top_tokens = [tokenizer.id2word[id] for id in top_indices_list[0]]

    return top_tokens

def run_predict():
    # 有 GPU 就用 CUDA，否则使用 CPU。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 推理必须加载训练时生成的词表，保证 token id 与模型权重对应。
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)
    print('vocabulary load success!')

    # 按词表大小创建模型，再加载训练过程中保存的最佳权重。
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))
    print('model load success!')

    print('Welcome to INTELEGER input method! print q or quit to exit...')
    # input_history 会累计用户连续输入的内容，模型基于完整历史预测后续词。
    input_history = ''
    while True:
        user_input = input('> ')
        if user_input.strip() in ['q', 'quit']:
            print('bye!')
            break
        elif user_input.strip() == '':
            print('please input valid content!')
            continue

        # 把新输入追加到历史上下文，再预测最可能的 5 个后续 token。
        input_history += user_input
        top_tokens = predict(input_history, model, tokenizer, 5, device)
        print('top tokens:', top_tokens)

if __name__ == '__main__':
    # 命令行直接执行 python src/predict.py 时进入交互式预测。
    run_predict()
