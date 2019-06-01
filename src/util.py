import json
import numpy as np
import re
import math
import random
import jieba
import joblib
from jieba import posseg as pseg
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


def pad_list(item_list, pad, max_len):
    item_list = item_list[:max_len]
    return item_list + [pad] * (max_len - len(item_list))


def pad_batch(data_batch, pad, max_len):
    return [pad_list(data, pad, max_len) for data in data_batch]


def convert_item(item, convert_dict, unk):
    return convert_dict[item] if item in convert_dict else unk


def convert_list(item_list, convert_dict, pad, unk, max_len=None):
    item_list = [convert_item(item, convert_dict, unk) for item in item_list]
    if max_len is not None:
        item_list = pad_list(item_list, pad, max_len)

    return item_list


def read_dict(dict_file):
    with open(dict_file, 'r', encoding='utf-8') as fin:
        key_2_id = json.load(fin)
        id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key


def cut_text(text, cut_all=False):
    text = re.sub(r'\s', '', text)
    return jieba.lcut(text, cut_all=cut_all)


def pos_text(text):
    text = re.sub(r'\s', '', text)
    return pseg.lcut(text)


def make_batch_iter(data, batch_size, shuffle):
    data_size = len(data)

    if shuffle:
        random.shuffle(data)

    num_batches = (data_size + batch_size - 1) // batch_size
    print('total batches: ', num_batches)
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(data_size, (i + 1) * batch_size)
        yield data[start_index: end_index]


def train_embedding(text_file, embedding_size, model_file):
    data = []
    with open(text_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append(line.strip().split())

    model = Word2Vec(data, size=embedding_size, window=5, min_count=5, workers=8)
    model.save(model_file)


def load_embedding(model_file, word_list):
    model = Word2Vec.load(model_file)

    embedding_matrix = []
    for word in word_list:
        if word in model:
            embedding_matrix.append(model[word])
        else:
            embedding_matrix.append(np.zeros(model.vector_size))

    return np.array(embedding_matrix)


def load_tencent_embedding(model_file, word_list):
    em = {}
    with open(model_file, 'r', encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            data = line.strip().split()
            if data[0] in word_list:
                em[data[0]] = list(map(float, data[1:]))

    embedding_matrix = []
    for word in word_list:
        embedding = em[word] if word in em else [0.0] * 200
        embedding_matrix.append(embedding)

    return np.array(embedding_matrix)


def train_tfidf(text_file, feature_size, model_file):
    with open(text_file, 'r', encoding='utf-8') as fin:
        data = fin.readlines()

    tfidf = TfidfVectorizer(
        max_features=feature_size,
        ngram_range=(1, 2),
        token_pattern=r'(?u)\b\w+\b'
    ).fit(data)

    joblib.dump(tfidf, model_file)


def load_tfidf(model_file):
    return joblib.load(model_file)


def load_gidf(model_file):
    gidf = {}
    with open(model_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            w, v = line.strip().split()
            gidf[w] = float(v)

    return gidf


def cosine_similarity(v1, v2):
    r = 0
    s1 = 0.0
    s2 = 0.0
    for x, y in zip(v1, v2):
        r += x * y
        s1 += x * x
        s2 += y * y
    return r / (math.sqrt(s1) * math.sqrt(s2) + 1e-10)


# ====================
def refine_text(text):
    text = re.sub(r'\s', '', text)

    text = re.sub(r'\d{4}年(\d{1,2}月份?)?(\d{1,2}日)?(\d{1,2}时)?(\d{1,2}分)?(许|左右)?', 'TIME', text)
    text = re.sub(r'同年\d{1,2}月份?(\d{1,2}日)?(\d{1,2}时)?(\d{1,2}分)?(许|左右)?', 'TIME', text)
    text = re.sub(r'同月\d{1,2}日(\d{1,2}时)?(\d{1,2}分)?(许|左右)?', 'TIME', text)
    text = re.sub(r'同日\d{1,2}时(\d{1,2}分)?(许|左右)?', 'TIME', text)
    text = re.sub(r'\d{1,2}时(\d{1,2}分)?(许|左右)?', 'TIME', text)

    text = re.sub(r'\d*([×xX])+\d*', 'UNK', text)

    text = jieba.lcut(text, cut_all=False)
    return text


def impr_2_id(time):
    # 将刑期用分类模型来做
    v = int(time['imprisonment'])

    if time['death_penalty']:
        return 0
    if time['life_imprisonment']:
        return 1
    elif v > 10 * 12:
        return 2
    elif v > 7 * 12:
        return 3
    elif v > 5 * 12:
        return 4
    elif v > 3 * 12:
        return 5
    elif v > 2 * 12:
        return 6
    elif v > 1 * 12:
        return 7
    else:
        return 8


def id_2_impr(y):
    # 返回每一个罪名区间的中位数
    if y == 0:
        return -2
    if y == 1:
        return -1
    if y == 2:
        return 120
    if y == 3:
        return 102
    if y == 4:
        return 72
    if y == 5:
        return 48
    if y == 6:
        return 30
    if y == 7:
        return 18
    else:
        return 6


def get_task_result(task_output, threshold):
    task_result = []
    for i, v in enumerate(task_output):
        if v >= threshold:
            task_result.append(i)
    return task_result
