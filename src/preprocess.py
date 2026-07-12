import pandas as pd
import jieba
from sklearn.model_selection import train_test_split

from config import *
from tokenizer import JiebaTokenizer

# 本文件负责把原始对话 jsonl 转换成模型可训练的 next-token 样本。
# 核心思路：先分词并转 token id，再用长度为 SEQ_LEN 的滑动窗口生成
# input=[前 SEQ_LEN 个 token]、target=下一个 token。

def build_dataset(sentences, tokenizer):
    # dataset 最终会写成 jsonl，每条记录包含 input 和 target 两个字段。
    dataset = []
    # 先把每个句子从文本编码成 token id 列表。
    sentences_id = [tokenizer.encode(sentence) for sentence in sentences ]
    for sentence_id_list in sentences_id:
        # 对一个句子的 token id 序列做滑动窗口。
        # 例如 SEQ_LEN=5 时，用位置 0-4 预测位置 5，用位置 1-5 预测位置 6。
        for i in range(len(sentence_id_list) - SEQ_LEN):
            input = sentence_id_list[i:i + SEQ_LEN]
            target = sentence_id_list[i + SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset

def preprocess():
    print('The preprocessing of data is starting...')

    # 原始文件是 jsonl，每一行是一段对话；sample(frac=0.1) 表示只抽取 10% 数据加快实验。
    df = pd.read_json(RAW_DATA_DIR/RAW_DATA_FILE, lines=True, orient='records').sample(frac=0.1)
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            # 原始对话形如 "user1：正文"，这里按中文全角冒号分割，只保留真正的正文部分。
            sentences.append(sentence.split('：')[1])
    # 打印一个样本和总句子数，便于确认预处理输入是否正常。
    print(sentences[0])
    print(len(sentences))

    # 固定 random_state，让训练集/测试集划分可复现。
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)

    # 只用训练集构建词表，避免测试集信息泄漏到训练阶段。
    JiebaTokenizer.build_vocab(train_sentences, MODEL_DIR/VOCAB_FILE)
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)

    # 分别生成训练集和测试集的滑动窗口样本。
    train_dataset = build_dataset(train_sentences, tokenizer)
    test_dataset = build_dataset(test_sentences, tokenizer)

    # 写出为 jsonl，后续 dataset.py 会逐行读取。
    pd.DataFrame(train_dataset).to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE, orient='records', lines=True)
    pd.DataFrame(test_dataset).to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE, orient='records', lines=True)

    print('The preprocessing of data is done.')

if __name__ == '__main__':
    # 命令行直接执行 python src/preprocess.py 时生成训练/测试数据和词表。
    preprocess()
