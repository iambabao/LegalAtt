import codecs
import json
import os
import random
import time
import numpy as np
import tensorflow as tf

from src import config
from src import util
from src.model import LawAtt


def pad_fact_batch(fact_batch):
    new_batch = []
    for fact in fact_batch:
        new_batch.append(util.pad_sequence(fact, config.SENTENCE_LEN, pad_type='id'))
    return new_batch


def inference(sess, model, batch_iter, kb_data, out_file, verbose=True):
    law_kb, law_len = kb_data

    task_1_output = []
    task_2_output = []
    task_3_output = []
    start_time = time.time()
    for i, batch in enumerate(batch_iter):
        if verbose:
            print('processing batch: %5d' % i, end='\r')

        fact, fact_len = list(zip(*batch))

        batch_size = len(fact)
        fact = pad_fact_batch(fact)

        feed_dict = {
            model.fact: fact,
            model.fact_len: fact_len,
            model.law_kb: [law_kb] * batch_size,
            model.law_len: [law_len] * batch_size
        }

        _task_1_output = sess.run(
            model.task_1_output,
            feed_dict=feed_dict
        )
        task_1_output.extend(_task_1_output)
        task_2_output.extend([[0.0] * config.ARTICLE_NUM] * batch_size)
        task_3_output.extend([[0.0] * config.IMPRISONMENT_NUM] * batch_size)
    print('\ncost time: %.3fs' % (time.time() - start_time))

    # 单标签
    # task_1_result = [[np.argmax(s, axis=-1)] for s in task_1_output]
    # task_2_result = [[np.argmax(s, axis=-1)] for s in task_2_output]
    # task_3_result = np.argmax(task_3_output, axis=-1)
    #
    # result = []
    # for t1, t2, t3 in zip(task_1_result, task_2_result, task_3_result):
    #     result.append({
    #         'accusation': t1,
    #         'articles': t2,
    #         'imprisonment': util.id_2_imprisonment(t3),
    #     })
    #
    # print('write file: ', out_file + '.json')
    # with codecs.open(out_file + '.json', 'w', encoding='utf-8') as f_out:
    #     for r in result:
    #         r = util.format_result(r)
    #         print(json.dumps(r), file=f_out)

    # 多标签
    for threshold in config.TASK_THRESHOLD:
        task_1_result = [util.get_task_result(s, threshold) for s in task_1_output]
        task_2_result = [util.get_task_result(s, threshold) for s in task_2_output]
        task_3_result = np.argmax(task_3_output, axis=-1)

        result = []
        for t1, t2, t3 in zip(task_1_result, task_2_result, task_3_result):
            result.append({
                'accusation': t1,
                'articles': t2,
                'imprisonment': util.id_2_imprisonment(t3),
            })

        print('write file: ', out_file + '-' + str(threshold) + '.json')
        with codecs.open(out_file + '-' + str(threshold) + '.json', 'w', encoding='utf-8') as f_out:
            for r in result:
                r = util.format_result(r)
                print(json.dumps(r), file=f_out)


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


def read_law_kb(data_dir, id_2_law, word_2_id, max_len):
    law_kb = []
    law_len = []
    for i in range(len(id_2_law)):
        law_name = id_2_law[i]
        file_name = os.path.join(data_dir, str(law_name) + '.txt')
        with codecs.open(file_name, 'r', encoding='utf-8') as f_in:
            law = f_in.readline()
            law = util.refine_text(law)
            law = util.convert_to_id_list(law, word_2_id)
            law = law[:max_len]

            law_len.append(len(law))

            law_kb.append(util.pad_sequence(law, max_len, pad_type='id'))

    return law_kb, law_len


def read_data(data_file, word_2_id, max_len):
    fact = []
    fact_len = []

    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            item = json.loads(line, encoding='utf-8')

            _fact = item['fact'].strip().lower()
            _fact = util.refine_text(_fact)
            _fact = util.convert_to_id_list(_fact, word_2_id)
            _fact = _fact[:max_len]
            fact.append(_fact)

            fact_len.append(len(_fact))

    print('data size: ', len(fact))

    return fact, fact_len


def predict(judger, config_proto):
    assert config.CURRENT_MODEL == 'law_att'

    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    if not os.path.exists(config.RESULT_DIR):
        os.makedirs(config.RESULT_DIR)

    word_2_id, id_2_word = util.read_dict(config.WORD_DICT)
    law_2_id, id_2_law, accu_2_id, id_2_accu = util.init_dict(config.LAW_DICT, config.ACCU_DICT)
    if os.path.exists(config.WORD2VEC_MODEL):
        embedding_matrix = util.load_embedding(config.WORD2VEC_MODEL, word_2_id.keys())
        embedding_trainable = False
    else:
        embedding_matrix = np.random.uniform(-0.5, 0.5, [config.VOCAB_SIZE, config.EMBEDDING_SIZE])
        embedding_trainable = True

    with tf.variable_scope('model', reuse=None):
        test_model = LawAtt(
            accu_num=config.ACCU_NUM, article_num=config.ARTICLE_NUM,
            top_k=config.TOP_K, max_seq_len=config.SENTENCE_LEN,
            hidden_size=config.HIDDEN_SIZE, att_size=config.ATT_SIZE,
            kernel_size=config.KERNEL_SIZE, filter_dim=config.FILTER_DIM, fc_size=config.FC_SIZE_S,
            embedding_matrix=embedding_matrix, embedding_trainable=embedding_trainable,
            lr=config.LR, optimizer=config.OPTIMIZER, keep_prob=config.KEEP_PROB, l2_rate=config.L2_RATE,
            is_training=False
        )

    test_data = read_data(config.TEST_DATA, word_2_id, max_len=config.SENTENCE_LEN)
    kb_data = read_law_kb(config.LAW_KB_DIR, id_2_law, word_2_id, max_len=config.SENTENCE_LEN)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=config_proto) as sess:
        print('load model from: ' + config.MODEL_FILE)
        saver.restore(sess, config.MODEL_FILE)

        print('==========  Test  ==========')
        test_batch_iter = make_batch_iter(list(zip(*test_data)), config.BATCH_SIZE, shuffle=False)
        inference(sess, test_model, test_batch_iter, kb_data, config.TEST_RESULT, verbose=True)

        # 单标签
        # result = judger.my_test(config.TEST_DATA, config.TEST_RESULT + '.json')
        # accu_micro_f1, accu_macro_f1 = judger.calc_f1(result[0])
        # article_micro_f1, article_macro_f1 = judger.calc_f1(result[1])
        # score = judger.get_score(result)
        # print('Threshold: %.3f' % threshold)
        # print('Micro-F1 of accusation: %.3f' % accu_micro_f1)
        # print('Macro-F1 of accusation: %.3f' % accu_macro_f1)
        # print('Micro-F1 of relevant articles: %.3f' % article_micro_f1)
        # print('Macro-F1 of relevant articles: %.3f' % article_macro_f1)
        # print('Score: ', score)

        # 多标签
        for threshold in config.TASK_THRESHOLD:
            result = judger.my_test(config.TEST_DATA, config.TEST_RESULT + '-' + str(threshold) + '.json')
            accu_micro_f1, accu_macro_f1 = judger.calc_f1(result[0])
            article_micro_f1, article_macro_f1 = judger.calc_f1(result[1])
            score = judger.get_score(result)
            print('Threshold: %.3f' % threshold)
            print('Micro-F1 of accusation: %.3f' % accu_micro_f1)
            print('Macro-F1 of accusation: %.3f' % accu_macro_f1)
            print('Micro-F1 of relevant articles: %.3f' % article_micro_f1)
            print('Macro-F1 of relevant articles: %.3f' % article_macro_f1)
            print('Score: ', score)
