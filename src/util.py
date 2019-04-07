import codecs
import json
import numpy as np
import re
import random
import jieba
from gensim.models import Word2Vec


def get_id(key, key_2_id, unk_id):
    return key_2_id[key] if key in key_2_id else unk_id


def get_key(i, id_2_key, unk):
    return id_2_key[i] if i in id_2_key else unk


def convert_to_key_list(id_list, id_2_key, pad, unk, max_len=None):
    key_list = [get_key(i, id_2_key, unk) for i in id_list]
    if max_len is not None:
        key_list = key_list[:max_len]
        key_list = key_list + [pad] * max(0, max_len - len(id_list))

    return key_list


def convert_to_id_list(key_list, key_2_id, pad_id, unk_id, max_len=None):
    id_list = [get_id(key, key_2_id, unk_id) for key in key_list]
    if max_len is not None:
        id_list = id_list[:max_len]
        id_list = id_list + [pad_id] * max(0, max_len - len(key_list))

    return id_list


def pad_sequence(seq, max_len, pad_item):
    seq = seq[:max_len]
    return seq + [pad_item] * max(0, max_len - len(seq))


def train_embedding(text_file, embedding_size, model_file):
    data = []
    with codecs.open(text_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append(line.strip().split())

    model = Word2Vec(data, size=embedding_size, window=5, min_count=5)
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


def read_dict(dict_file):
    with codecs.open(dict_file, 'r', encoding='utf-8') as fin:
        key_2_id = json.load(fin)
        id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key


def refine_text(text):
    text = re.sub(r'\s', '', text)

    text = re.sub(r'\d{4}年(\d{1,2}月份?)?(\d{1,2}日)?(\d{1,2}时)?(\d{1,2}分)?(许|左右)?', 'TIME', text)
    text = re.sub(r'同年\d{1,2}月份?(\d{1,2}日)?(\d{1,2}时)?(\d{1,2}分)?(许|左右)?', 'TIME', text)
    text = re.sub(r'同月\d{1,2}日(\d{1,2}时)?(\d{1,2}分)?(许|左右)?', 'TIME', text)
    text = re.sub(r'同日\d{1,2}时(\d{1,2}分)?(许|左右)?', 'TIME', text)

    text = re.sub(r'\d*[xX]+\d*', 'PAD', text)

    text = jieba.lcut(text, cut_all=False)
    return text


# ----------
# 以下根据具体任务改


# convert word list to sequence list, each sequence contains one or more sentence.
def refine_doc(data, max_seq_len, max_doc_len):
    doc = []
    beg, end = 0, 0
    while end < len(data):
        end += 1
        if data[end - 1] in ['，', '。', '！', '？', '；']:
            doc.append(data[beg:min(end, beg + max_seq_len)])
            beg = end
    if beg != end:
        doc.append(data[beg:min(end, beg + max_seq_len)])
    beg = 0
    while len(doc) > max_doc_len and beg < len(doc) - 1:
        if len(doc[beg]) + len(doc[beg + 1]) <= max_seq_len:
            doc[beg].extend(doc[beg + 1])
            del doc[beg + 1]
        else:
            beg += 1

    doc = doc[:max_doc_len]
    return doc


def init_dict(law_dict, accu_dict):
    f = open(law_dict, 'r', encoding='utf-8')
    law_2_id = {}
    id_2_law = {}
    line = f.readline()
    while line:
        id_2_law[len(law_2_id)] = line.strip()
        law_2_id[line.strip()] = len(law_2_id)
        line = f.readline()
    f.close()

    f = open(accu_dict, 'r', encoding='utf-8')
    accu_2_id = {}
    id_2_accu = {}
    line = f.readline()
    while line:
        id_2_accu[len(accu_2_id)] = line.strip()
        accu_2_id[line.strip()] = len(accu_2_id)
        line = f.readline()
    f.close()

    return law_2_id, id_2_law, accu_2_id, id_2_accu


def imprisonment_2_id(time):
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


def id_2_imprisonment(y):
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
            task_result.append(i + 1)
    return task_result
