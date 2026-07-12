import jieba
from config import *

# 本文件封装 jieba 分词器，并维护 token 与 id 之间的双向映射。
# 模型只能处理数字 id，所以所有文本都需要先经过这里编码。

class JiebaTokenizer():
    # 类属性：所有实例共享同一个未知词标记。
    unk_token = UNK_TOKEN
    def __init__(self, vocab_list):
        # vocab_list 的顺序决定 token id；训练、评估、推理必须使用同一份词表文件。
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        # word2id 用于编码文本，id2word 用于把预测结果还原成可读 token。
        self.word2id = {word : id for id, word in enumerate(vocab_list)}
        self.id2word = {id : word for id, word in enumerate(vocab_list)}
        # 未登录词统一映射到 <unk> 的 id，避免模型遇到新词时报错。
        self.unk_token_id = self.word2id[self.unk_token]

    @staticmethod
    def tokenize(text):
        # jieba.lcut 会直接返回分词后的 list[str]。
        return jieba.lcut(text)

    def encode(self, text):
        # 先分词，再把每个 token 转成对应 id；词表没有的 token 使用 <unk>。
        tokens = self.tokenize(text)
        ids = [self.word2id.get(token, self.unk_token_id) for token in tokens]
        return ids

    @classmethod
    def build_vocab(cls, sentences, vocab_file_path):
        # 用 set 去重，收集训练语料中出现过的所有 token。
        vocab_set = set()
        for sentence in sentences:
            vocab_set.update(jieba.lcut(sentence))
        # <unk> 放在词表第一位，保证未知词有固定 id。
        vocab_list = [cls.unk_token] + list(vocab_set)

        print("The size of vocabulary:", len(vocab_list))

        # 每行写一个 token，from_vocab 会按相同顺序读回来。
        with open(vocab_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_file_path):
        # 从磁盘加载词表，重建 tokenizer。
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            vocab_list = [token.strip() for token in f.readlines()]
        tokenizer = cls(vocab_list)
        return tokenizer

if __name__ == '__main__':
    # 单独运行本文件时，检查词表加载和编码效果。
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)
    print('the size of vocabulary:', len(tokenizer.vocab_list))
    print('UNK_TOKEN: ', tokenizer.unk_token)
    print(tokenizer.encode('自然语言处理'))
