from pathlib import Path

# 项目根目录：config.py 位于 src 目录下，所以 parent.parent 会回到项目根目录。
ROOT_DIR = Path(__file__).parent.parent

# 原始数据目录和预处理后数据目录。
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'

# 训练产物目录：模型权重、词表和 TensorBoard 日志都会从这里读写。
MODEL_DIR = ROOT_DIR / 'models'
LOG_DIR = ROOT_DIR / 'logs'

# 各阶段会用到的文件名。
RAW_DATA_FILE = 'synthesized_.jsonl'
TRAIN_DATA_FILE = 'train.jsonl'
TEST_DATA_FILE = 'test.jsonl'
VOCAB_FILE = 'vocab.txt'
BEST_MODEL = 'best_model.pt'

# 未登录词 token：分词结果不在词表里时统一映射到这个占位符。
UNK_TOKEN = '<unk>'

# 模型和训练超参数。
# SEQ_LEN 表示用前 5 个 token 预测第 6 个 token。
SEQ_LEN = 5
BATCH_SIZE = 64
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256

# 优化器学习率和训练轮数。
LEARNING_RATE = 0.001
EPOCHS = 40
