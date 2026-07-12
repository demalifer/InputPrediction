import pandas as pd
import jieba
from sklearn.model_selection import train_test_split

from config import *
from tokenizer import JiebaTokenizer

def build_dataset(sentences, tokenizer):
    dataset = []
    sentences_id = [tokenizer.encode(sentence) for sentence in sentences ]
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

    JiebaTokenizer.build_vocab(train_sentences, MODEL_DIR/VOCAB_FILE)
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)

    train_dataset = build_dataset(train_sentences, tokenizer)
    test_dataset = build_dataset(test_sentences, tokenizer)

    pd.DataFrame(train_dataset).to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE, orient='records', lines=True)
    pd.DataFrame(test_dataset).to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE, orient='records', lines=True)

    print('The preprocessing of data is done.')

if __name__ == '__main__':
    preprocess()