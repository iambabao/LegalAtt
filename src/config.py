import os

# 根目录
ROOT_DIR = os.path.abspath('.')

# 存储临时文件
TEMP_DIR = os.path.join(ROOT_DIR, 'temp')

# 存储数据文件
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_DATA = os.path.join(DATA_DIR, 'data_train.json')
VALID_DATA = os.path.join(DATA_DIR, 'data_valid.json')
TEST_DATA = os.path.join(DATA_DIR, 'data_test.json')
WORD_DICT = os.path.join(DATA_DIR, 'word_dict.json')
WORD_COUNT = os.path.join(DATA_DIR, 'word_count.json')
ACCU_DICT = os.path.join(DATA_DIR, 'accu.txt')
LAW_DICT = os.path.join(DATA_DIR, 'law.txt')

# 存储法律文本
LAW_KB_DIR = os.path.join(DATA_DIR, 'law_kb')

# 存储词向量文件
EMBEDDING_DIR = os.path.join(DATA_DIR, 'embedding')
PLAIN_TEXT = os.path.join(EMBEDDING_DIR, 'plain_text.txt')
WORD2VEC_MODEL = os.path.join(EMBEDDING_DIR, 'word2vec.model')
TFIDF_MODEL = os.path.join(EMBEDDING_DIR, 'tfidf.model')

CURRENT_MODEL = 'law_att'

# 存储训练模型
MODEL_DIR = os.path.join(ROOT_DIR, 'model', CURRENT_MODEL)
MODEL_FILE = os.path.join(MODEL_DIR, 'model')

# 存储预测结果
RESULT_DIR = os.path.join(ROOT_DIR, 'result', CURRENT_MODEL)
VALID_RESULT = os.path.join(RESULT_DIR, 'valid_result')
TEST_RESULT = os.path.join(RESULT_DIR, 'test_result')

GPU_ID = '0'  # '0, 1'指定多块显卡

PAD = 'PAD'
PAD_ID = 0
UNK = 'UNK'
UNK_ID = 1
NUM = 'NUM'
NUM_ID = 2
VOCAB_SIZE = 100000

ACCU_NUM = 202
ARTICLE_NUM = 183
IMPRISONMENT_NUM = 9

TASK_THRESHOLD = [0.1, 0.2, 0.3, 0.4, 0.5]

# LinearSVM
TFIDF_SIZE = 5000

# RNN
HIDDEN_SIZE = 100

# CNN
KERNEL_SIZE = [2, 3, 4, 5]
FILTER_DIM = 50

# Transformer
BLOCK_NUM = 5
HEAD_NUM = 5
MODEL_DIM = 200

# ATT
ATT_SIZE = 100

# FC
FC_SIZE_S = 200
FC_SIZE_M = 500
FC_SIZE_L = 1000

# Fact_law
TOP_K = 5

NUM_EPOCH = 30
BATCH_SIZE = 64
SENTENCE_LEN = 500
SEQUENCE_LEN = 100
DOCUMENT_LEN = 20
EMBEDDING_SIZE = 200

OPTIMIZER = 'Adam'
LR = 0.001
KEEP_PROB = 0.8
L2_RATE = 0.0
