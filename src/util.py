import codecs
import json
import numpy as np
import re
import jieba
from gensim.models import Word2Vec

from src import config


def get_id(key, key_2_id):
    return key_2_id[key] if key in key_2_id else config.UNK_ID


def get_key(i, id_2_key):
    return id_2_key[i] if i in id_2_key else config.UNK


def convert_to_key_list(id_list, id_2_key, max_len=None):
    key_list = [get_key(i, id_2_key) for i in id_list]
    if max_len is not None:
        key_list = key_list[:max_len]
        key_list = key_list + [config.PAD] * max(0, max_len - len(id_list))

    return key_list


def convert_to_id_list(key_list, key_2_id, max_len=None):
    id_list = [get_id(key, key_2_id) for key in key_list]
    if max_len is not None:
        id_list = id_list[:max_len]
        id_list = id_list + [config.PAD_ID] * max(0, max_len - len(key_list))

    return id_list


def pad_sequence(seq, max_len, pad_type):
    seq = seq[:max_len]
    pad = config.PAD_ID if pad_type == 'id' else config.PAD
    return seq + [pad] * max(0, max_len - len(seq))


def train_embedding(text_file, embedding_size, model_file):
    data = []
    with codecs.open(text_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
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

    return np.asarray(embedding_matrix)


def read_dict(dict_file):
    with codecs.open(dict_file, 'r', encoding='utf-8') as f_in:
        key_2_id = json.load(f_in)
        id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key

# ----------
# 以下根据具体任务改


# 预处理加分词
def refine_text(text):
    text = text.replace(' ', '')
    text = re.sub(r'[\r\n\t]', '', text)

    # 将1,000元的格式转化为1000元的格式
    it = re.finditer(r'([1-9][0-9]*([,，][0-9]{3})+(\.[0-9]*)?)([千万亿]*余?[千万亿]*元)', text)
    for match in it:
        t = match.group()
        new_t = re.sub(r'[,，]', '', t)
        text = text.replace(t, new_t)

    # 将以元为单位的金额按1000元划分，向下取整，如0-999元替换为0元，1000-1999元替换为1000元
    it = re.finditer(r'([1-9][0-9]*(\.[0-9]*)?)([余]*元)', text)
    for match in it:
        num = eval(match.group(1))
        unit = match.group(3)
        text = text.replace(str(num) + unit, str(int(num / 1000) * 1000) + unit)

    # 将以千元为单位的金额按10千元划分，向下取整。
    it = re.finditer(r'([1-9][0-9]*(\.[0-9]*)?)([余]*千余?元)', text)
    for match in it:
        num = eval(match.group(1))
        unit = match.group(3)
        text = text.replace(str(num) + unit, str(int(num / 10) * 10) + unit)

    # 将以万元为单位的金额按1万元划分，向下取整。
    it = re.finditer(r'([1-9][0-9]*(\.[0-9]*)?)([千余]*万余?元)', text)
    for match in it:
        num = eval(match.group(1))
        unit = match.group(3)
        text = text.replace(str(num) + unit, str(int(num)) + unit)

    # 将以亿元为单位的金额按1亿元划分，向下取整。
    it = re.finditer(r'([1-9][0-9]*(\.[0-9]*)?)([千万余]*亿余?元)', text)
    for match in it:
        num = eval(match.group(1))
        unit = match.group(3)
        text = text.replace(str(num) + unit, str(int(num)) + unit)

    text = [_ for _ in jieba.cut(text, cut_all=False)]
    return text


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


# Format the result generated by the Predictor class
def format_result(result):
    rex = {"accusation": [], "articles": [], "imprisonment": -3}

    res_acc = []
    for x in result["accusation"]:
        if not (x is None):
            res_acc.append(int(x))
    rex["accusation"] = res_acc

    if not (result["imprisonment"] is None):
        rex["imprisonment"] = int(result["imprisonment"])
    else:
        rex["imprisonment"] = -3

    res_art = []
    for x in result["articles"]:
        if not (x is None):
            res_art.append(int(x))
    rex["articles"] = res_art

    return rex
