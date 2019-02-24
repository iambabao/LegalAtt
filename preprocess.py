import os
import codecs
import json
import collections
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.externals import joblib

from src import config
from src import util


def get_data_text(data_file, text_file, to_lower=True):
    f_out = codecs.open(text_file, 'a+', encoding='utf-8')

    counter = 0
    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            data = json.loads(line, encoding='utf-8')
            fact = data['fact'].strip()
            if to_lower:
                fact = fact.lower()
            fact = util.refine_text(fact)
            print(' '.join(fact), file=f_out)
            counter += 1
            if counter % 10000 == 0:
                print('processing: ', counter)

    f_out.close()


def get_kb_text(kb_dir, text_file, to_lower=True):
    f_out = codecs.open(text_file, 'a+', encoding='utf-8')

    counter = 0
    print('read dir: ', kb_dir)
    file_list = os.listdir(kb_dir)
    for file in file_list:
        file_name = os.path.join(kb_dir, file)
        with codecs.open(file_name, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if to_lower:
                    line = line.lower()
                line = util.refine_text(line)
                print(' '.join(line), file=f_out)
                counter += 1
                if counter % 10000 == 0:
                    print('processing: ', counter)

    f_out.close()


def build_word_dict(data_file, word_dict_file, word_count_file, vocab_size, to_lower=True):
    counter = collections.Counter()

    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            for word in line.strip().split():
                if to_lower:
                    word = word.lower()
                counter[word] += 1
    print('number of words: ', len(counter))

    del counter[' ']
    del counter[config.PAD]
    del counter[config.UNK]
    del counter[config.NUM]
    sorted_counter = dict(sorted(
        counter.items(),
        key=itemgetter(1),
        reverse=True
    ))
    del counter

    print('write file: ', word_count_file)
    with codecs.open(word_count_file, 'w', encoding='utf-8') as f_out:
        json.dump(sorted_counter, f_out, ensure_ascii=False, indent=4)

    word_2_id = {config.PAD: config.PAD_ID, config.UNK: config.UNK_ID, config.NUM: config.NUM_ID}
    for word in sorted_counter.keys():
        word_2_id[word] = len(word_2_id)
        if len(word_2_id) >= vocab_size:
            break

    print('write file: ', word_dict_file)
    with codecs.open(word_dict_file, 'w', encoding='utf-8') as f_out:
        json.dump(word_2_id, f_out, ensure_ascii=False, indent=4)


def train_tfidf(data_file, feature_size, model_file):
    with codecs.open(data_file, 'r', encoding='utf-8') as f_in:
        data = f_in.readlines()

    tfidf = TFIDF(
        max_features=feature_size,
        ngram_range=(1, 3)
    )
    tfidf.fit(data)

    joblib.dump(tfidf, model_file)


def preprocess():
    if not os.path.exists(config.EMBEDDING_DIR):
        os.makedirs(config.EMBEDDING_DIR)

    print('extract plain text')
    get_data_text(config.TRAIN_DATA, config.PLAIN_TEXT, to_lower=True)
    get_data_text(config.VALID_DATA, config.PLAIN_TEXT, to_lower=True)
    get_data_text(config.TEST_DATA, config.PLAIN_TEXT, to_lower=True)
    get_kb_text(config.LAW_KB_DIR, config.PLAIN_TEXT, to_lower=True)

    print('build word dict')
    build_word_dict(config.PLAIN_TEXT, config.WORD_DICT, config.WORD_COUNT, config.VOCAB_SIZE, to_lower=True)

    print('train word embedding')
    util.train_embedding(config.PLAIN_TEXT, config.EMBEDDING_SIZE, config.WORD2VEC_MODEL)

    print('train tfidf')
    train_tfidf(config.PLAIN_TEXT, config.TFIDF_SIZE, config.TFIDF_MODEL)


if __name__ == '__main__':
    preprocess()
