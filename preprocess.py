import os
import codecs
import json
import collections
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.externals import joblib

from src.config import Config
from src import util


def get_data_text(data_file, text_file, to_lower=True):
    f_out = codecs.open(text_file, 'a+', encoding='utf-8')

    counter = 0
    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            fact = json.loads(line)['fact']
            fact = fact.strip()
            if to_lower:
                fact = fact.lower()
            print(fact, file=f_out)
            counter += 1
            if counter % 10000 == 0:
                print('processing: ', counter)

    f_out.close()


def build_word_dict(data_file, word_count_file, word_dict_file, vocab_size, to_lower=True):
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
    del counter[config.pad]
    del counter[config.unk]
    del counter[config.num]
    sorted_counter = dict(sorted(
        counter.items(),
        key=itemgetter(1),
        reverse=True
    ))
    del counter

    print('write file: ', word_count_file)
    with codecs.open(word_count_file, 'w', encoding='utf-8') as f_out:
        json.dump(sorted_counter, f_out, ensure_ascii=False, indent=4)

    word_2_id = {config.pad: config.pad_id, config.unk: config.unk_id, config.num: config.num_id}
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


def preprocess(config):
    if not os.path.exists(config.embedding_dir):
        os.makedirs(config.embedding_dir)

    print('extract plain text')
    get_data_text(config.train_data, config.plain_text, to_lower=True)
    get_data_text(config.valid_data, config.plain_text, to_lower=True)
    get_data_text(config.test_data, config.plain_text, to_lower=True)

    print('build word dict')
    build_word_dict(config.plain_text, config.word_count, config.word_dict, vocab_size=100000, to_lower=True)

    print('train word embedding')
    util.train_embedding(config.plain_text, config.embedding_size, config.word2vec_model)

    print('train tfidf')
    train_tfidf(config.plain_text, config.tfidf_size, config.tfidf_model)


if __name__ == '__main__':
    config = Config('/home/babao/001_workspaces/001_python/charge_new/', 'temp')
    preprocess(config)
