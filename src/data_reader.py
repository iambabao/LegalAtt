import codecs
import json
import os
from sklearn.externals import joblib

from src import util


def read_data(data_file, word_2_id, accu_2_id, law_2_id, config):
    fact = []
    fact_len = []
    accu = []
    article = []
    imprisonment = []

    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            item = json.loads(line, encoding='utf-8')

            _fact = item['fact'].strip().split()
            _fact = util.convert_to_id_list(_fact, word_2_id, config.pad_id, config.unk_id)
            _fact = _fact[:config.sentence_len]
            fact.append(_fact)

            fact_len.append(len(_fact))

            temp = item['meta']['accusation']
            for i in range(len(temp)):
                temp[i] = temp[i].replace('[', '').replace(']', '')
            temp = [accu_2_id[v] for v in temp]
            _accu = [0] * config.accu_num
            for i in temp:
                _accu[i] = 1
            accu.append(_accu)

            temp = [str(t) for t in item['meta']['relevant_articles']]
            temp = [law_2_id[v] for v in temp]
            _article = [0] * config.article_num
            for i in temp:
                _article[i] = 1
            article.append(_article)

            temp = item['meta']['term_of_imprisonment']
            temp = util.imprisonment_2_id(temp)
            _imprisonment = [0] * config.imprisonment_num
            _imprisonment[temp] = 1
            imprisonment.append(_imprisonment)

    print('data size: ', len(fact))

    return fact, fact_len, accu, article, imprisonment


def read_data_doc(data_file, word_2_id, accu_2_id, law_2_id, config):
    fact = []
    fact_seq_len = []
    fact_doc_len = []
    accu = []
    article = []
    imprisonment = []

    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            item = json.loads(line, encoding='utf-8')

            _fact = item['fact'].strip().split()
            _fact = util.refine_doc(_fact, config.sequence_len, config.document_len)
            _fact = [util.convert_to_id_list(seq, word_2_id, config.pad_id, config.unk_id) for seq in _fact]
            fact.append(_fact)

            _fact_seq_len = [0] * config.document_len
            for i in range(len(_fact)):
                _fact_seq_len[i] = len(_fact[i])
            fact_seq_len.append(_fact_seq_len)

            fact_doc_len.append(len(_fact))

            temp = item['meta']['accusation']
            for i in range(len(temp)):
                temp[i] = temp[i].replace('[', '').replace(']', '')
            temp = [accu_2_id[v] for v in temp]
            _accu = [0] * config.accu_num
            for i in temp:
                _accu[i] = 1
            accu.append(_accu)

            temp = [str(t) for t in item['meta']['relevant_articles']]
            temp = [law_2_id[v] for v in temp]
            _article = [0] * config.article_num
            for i in temp:
                _article[i] = 1
            article.append(_article)

            temp = item['meta']['term_of_imprisonment']
            temp = util.imprisonment_2_id(temp)
            _imprisonment = [0] * config.imprisonment_num
            _imprisonment[temp] = 1
            imprisonment.append(_imprisonment)

    print('data size: ', len(fact))

    return fact, fact_seq_len, fact_doc_len, accu, article, imprisonment


def read_data_doc_v2(data_file, word_2_id, accu_2_id, law_2_id, config):
    corpus = []
    fact = []
    fact_seq_len = []
    fact_doc_len = []
    accu = []
    article = []
    imprisonment = []

    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            item = json.loads(line, encoding='utf-8')

            _fact = item['fact'].strip().split()
            corpus.append(' '.join(_fact))

            _fact = util.refine_doc(_fact, config.sequence_len, config.document_len)
            _fact = [util.convert_to_id_list(seq, word_2_id, config.pad_id, config.unk_id) for seq in _fact]
            fact.append(_fact)

            _fact_seq_len = [0] * config.document_len
            for i in range(len(_fact)):
                _fact_seq_len[i] = len(_fact[i])
            fact_seq_len.append(_fact_seq_len)

            fact_doc_len.append(len(_fact))

            temp = item['meta']['accusation']
            for i in range(len(temp)):
                temp[i] = temp[i].replace('[', '').replace(']', '')
            temp = [accu_2_id[v] for v in temp]
            _accu = [0] * config.accu_num
            for i in temp:
                _accu[i] = 1
            accu.append(_accu)

            temp = [str(t) for t in item['meta']['relevant_articles']]
            temp = [law_2_id[v] for v in temp]
            _article = [0] * config.article_num
            for i in temp:
                _article[i] = 1
            article.append(_article)

            temp = item['meta']['term_of_imprisonment']
            temp = util.imprisonment_2_id(temp)
            _imprisonment = [0] * config.imprisonment_num
            _imprisonment[temp] = 1
            imprisonment.append(_imprisonment)

    print('data size: ', len(fact))
    tfidf_model = joblib.load(config.tfidf_model)

    return fact, fact_seq_len, fact_doc_len, tfidf_model.transform(corpus).toarray(), accu, article, imprisonment


def read_law_kb(id_2_law, word_2_id, config):
    law_kb = []
    law_len = []
    for i in range(len(id_2_law)):
        law_name = id_2_law[i]
        file_name = os.path.join(config.law_kb_dir, str(law_name) + '.txt')
        with codecs.open(file_name, 'r', encoding='utf-8') as fin:
            law = fin.readline()
            law = util.refine_text(law)
            law = util.convert_to_id_list(law, word_2_id, config.pad_id, config.unk_id)
            law = law[:config.sentence_len]

            law_len.append(len(law))

            law_kb.append(util.pad_sequence(law, config.sentence_len, pad_item=config.pad_id))

    return law_kb, law_len


def read_law_kb_doc(id_2_law, word_2_id, config):
    law_kb = []
    law_seq_len = []
    law_doc_len = []
    for i in range(len(id_2_law)):
        law_name = id_2_law[i]
        file_name = os.path.join(config.law_kb_dir, str(law_name) + '.txt')
        with codecs.open(file_name, 'r', encoding='utf-8') as fin:
            law = fin.readline()
            law = util.refine_text(law)
            law = util.refine_doc(law, config.sequence_len, config.document_len)
            law = [util.convert_to_id_list(seq, word_2_id, config.pad_id, config.unk_id) for seq in law]
            law_kb.append(law)

            seq_len = [0] * config.document_len
            for j in range(len(law)):
                seq_len[j] = len(law[j])
            law_seq_len.append(seq_len)

            law_doc_len.append(len(law))

    return pad_law_kb_doc(law_kb, config), law_seq_len, law_doc_len


def pad_fact_batch(fact_batch, config):
    new_batch = []
    for fact in fact_batch:
        new_batch.append(util.pad_sequence(fact, config.sentence_len, pad_item=config.pad_id))
    return new_batch


def pad_fact_batch_doc(fact_batch, config):
    new_batch = []
    for fact in fact_batch:
        temp = [[config.pad_id] * config.sequence_len] * config.document_len
        for i in range(len(fact)):
            temp[i][:len(fact[i])] = fact[i]
        new_batch.append(temp)
    return new_batch


def pad_law_kb_doc(law_kb, config):
    new_law_kb = []
    for art in law_kb:
        temp = [[config.pad_id] * config.sequence_len] * config.document_len
        for i in range(len(art)):
            temp[i][:len(art[i])] = art[i]
        new_law_kb.append(temp)
    return new_law_kb
