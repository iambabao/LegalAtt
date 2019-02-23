import codecs
import json
import os
import time
import random
import numpy as np
import tensorflow as tf

from src import config
from src import util
from src.model import DPCNN


def inference(sess, model, batch_iter, out_file, verbose=True):
    task_1_output = []
    task_2_output = []
    task_3_output = []
    start_time = time.time()
    for i, batch in enumerate(batch_iter):
        if verbose:
            print('processing batch: %5d' % i, end='\r')

        feed_dict = {
            model.fact: batch
        }

        _task_1_output, _task_2_output, _task_3_output = sess.run(
            [model.task_1_output, model.task_2_output, model.task_3_output],
            feed_dict=feed_dict
        )
        task_1_output.extend(_task_1_output)
        task_2_output.extend(_task_2_output)
        task_3_output.extend(_task_3_output)
    print('\ncost time: %.3fs' % (time.time() - start_time))

    # 单标签
    # task_1_result = [[np.argmax(s, axis=-1)] for s in task_1_output]
    # task_2_result = np.argmax(task_2_output, axis=-1)
    # task_3_result = [[np.argmax(s, axis=-1)] for s in task_3_output]
    #
    # result = []
    # for t1, t2, t3 in zip(task_1_result, task_2_result, task_3_result):
    #     result.append({
    #         'articles': t1,
    #         'imprisonment': util.id_2_imprisonment(t2),
    #         'accusation': t3
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
        task_2_result = np.argmax(task_2_output, axis=-1)
        task_3_result = [util.get_task_result(s, threshold) for s in task_3_output]

        result = []
        for t1, t2, t3 in zip(task_1_result, task_2_result, task_3_result):
            result.append({
                'articles': t1,
                'imprisonment': util.id_2_imprisonment(t2),
                'accusation': t3
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


def read_data(data_file, word_2_id, max_len):
    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    print('data size: ', len(lines))

    fact = []
    for line in lines:
        item = json.loads(line, encoding='utf-8')

        _fact = item['fact'].strip().lower()
        _fact = util.refine_text(_fact)
        _fact = util.convert_to_id_list(_fact, word_2_id, max_len=max_len)
        fact.append(_fact)

    return fact


def predict(judger, config_proto):
    assert config.CURRENT_MODEL == 'dpcnn'

    word_2_id, id_2_word = util.read_dict(config.WORD_DICT)
    # law_2_id, id_2_law, accu_2_id, id_2_accu = util.init_dict(config.LAW_DICT, config.ACCU_DICT)
    if os.path.exists(config.WORD2VEC_MODEL):
        embedding_matrix = util.load_embedding(config.WORD2VEC_MODEL, word_2_id.keys())
        embedding_trainable = False
    else:
        embedding_matrix = np.random.uniform(-0.5, 0.5, [config.VOCAB_SIZE, config.EMBEDDING_SIZE])
        embedding_trainable = True

    with tf.variable_scope('model', reuse=None):
        test_model = DPCNN(
            accu_num=config.ACCU_NUM, article_num=config.ARTICLE_NUM, imprisonment_num=config.IMPRISONMENT_NUM,
            seq_len=config.SENTENCE_LEN, filter_size=config.FILTER_SIZE[0],
            filter_dim=config.FILTER_DIM, fc_size=config.FC_SIZE_S,
            embedding_matrix=embedding_matrix, embedding_trainable=embedding_trainable,
            lr=config.LR, optimizer=config.OPTIMIZER, keep_prob=config.KEEP_PROB, l2_rate=config.L2_RATE,
            is_training=False
        )

    test_data = read_data(config.TEST_DATA, word_2_id, max_len=config.SENTENCE_LEN)

    saver = tf.train.Saver()
    with tf.Session(config=config_proto) as sess:
        print('load model from: ' + config.MODEL_FILE)
        saver.restore(sess, config.MODEL_FILE)

        print('==========  Test  ==========')
        test_batch_iter = make_batch_iter(test_data, config.BATCH_SIZE, shuffle=False)
        inference(sess, test_model, test_batch_iter, config.TEST_RESULT, verbose=True)

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
