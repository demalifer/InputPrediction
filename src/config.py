from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'

MODEL_DIR = ROOT_DIR / 'models'
LOG_DIR = ROOT_DIR / 'logs'

RAW_DATA_FILE = 'synthesized_.jsonl'
TRAIN_DATA_FILE = 'train.jsonl'
TEST_DATA_FILE = 'test.jsonl'
VOCAB_FILE = 'vocab.txt'
BEST_MODEL = 'best_model.pt'

UNK_TOKEN = '<unk>'

SEQ_LEN = 5
BATCH_SIZE = 64
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256

LEARNING_RATE = 0.001
EPOCHS = 40
