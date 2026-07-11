import pandas as pd
import jieba
from sklearn.model_selection import train_test_split

from config import *

def build_dataset(sentences, word2id):
    dataset = []
    sentences_id = [[word2id.get(token, 0) for token in jieba.lcut(sentence)] for sentence in sentences ]
    for sentence_id_list in sentences_id:
        for i in range(len(sentence_id_list) - SEQ_LEN):
            input = sentence_id_list[i:i + SEQ_LEN]
            target = sentence_id_list[i + SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset

def preprocess():
    print('The preprocessing of data is starting...')

    df = pd.read_json(RAW_DATA_DIR/RAW_DATA_FILE, lines=True, orient='records').sample(frac=0.1)
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split('：')[1])
    print(sentences[0])
    print(len(sentences))

    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)

    vocab_set = set()
    for sentence in train_sentences:
        vocab_set.update(jieba.lcut(sentence))
    vocab_list = [UNK_TOKEN] + list(vocab_set)
    word2id = {word : id for id, word in enumerate(vocab_list)}

    print('The size of vocabulary:', len(vocab_list))

    with open(MODEL_DIR/VOCAB_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_list))

    train_dataset = build_dataset(train_sentences, word2id)
    test_dataset = build_dataset(test_sentences, word2id)

    pd.DataFrame(train_dataset).to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE, orient='records', lines=True)
    pd.DataFrame(test_dataset).to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE, orient='records', lines=True)

    print('The preprocessing of data is done.')

if __name__ == '__main__':
    preprocess()