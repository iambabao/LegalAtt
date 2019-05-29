import json
import collections

from src.config import Config
from src import util


def build_word_dict(data_file, dict_file, vocab_size):
    counter = collections.Counter()

    with open(data_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            for word in line.strip().split():
                counter[word] += 1
    counter['PAD'] = 999999999
    counter['UNK'] = 999999998
    counter['NUM'] = 999999997
    counter['TIME'] = 999999996
    print('number of words: ', len(counter))

    word_dict = {}
    for word, _ in counter.most_common(vocab_size):
        word_dict[word] = len(word_dict)

    with open(dict_file, 'w', encoding='utf-8') as fout:
        json.dump(word_dict, fout, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    config = Config('./', 'temp')

    print('build word dict...')
    build_word_dict(config.plain_text, config.word_dict, config.vocab_size)

    print('train word embedding...')
    util.train_embedding(config.plain_text, config.embedding_size, config.word2vec_model)

    print('train tfidf...')
    util.train_tfidf(config.plain_text, config.tfidf_size, config.tfidf_model)
