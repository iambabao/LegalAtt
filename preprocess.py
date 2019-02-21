import os
import codecs
import json
import collections
from operator import itemgetter

from src import config
from src import util


def get_plain_text(data_file, text_file, word_2_id):
    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    f_out = codecs.open(text_file, 'w', encoding='utf-8')

    print('write file: ', text_file)
    counter = 0
    for line in lines:
        data = json.loads(line, encoding='utf-8')
        fact = data['fact'].strip().lower()
        fact = util.refine_text(fact)
        for i, w in enumerate(fact):
            if w not in word_2_id:
                fact[i] = config.UNK
        print(' '.join(fact), file=f_out)
        counter += 1
        if counter % 10000 == 0:
            print('processing: ', counter)

    f_out.close()


def build_word_dict(data_file, word_dict_file, word_count_file, vocab_size, to_lower=True):
    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    print('data size: ', len(lines))

    counter = collections.Counter()
    for line in lines:
        data = json.loads(line, encoding='utf-8')
        fact = data['fact'].strip().lower()
        fact = util.refine_text(fact)
        for word in fact:
            if to_lower:
                word = word.lower()
            counter[word] += 1
    print('number of words: ', len(counter))
    del lines

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

    return word_2_id


def preprocess():
    if not os.path.exists(config.EMBEDDING_DIR):
        os.makedirs(config.EMBEDDING_DIR)

    word_2_id = build_word_dict(config.TRAIN_DATA, config.WORD_DICT, config.WORD_COUNT, config.VOCAB_SIZE)
    get_plain_text(config.TRAIN_DATA, config.PLAIN_TEXT, word_2_id)
    util.train_embedding(config.PLAIN_TEXT, config.EMBEDDING_SIZE, config.WORD2VEC_MODEL)


if __name__ == '__main__':
    preprocess()
