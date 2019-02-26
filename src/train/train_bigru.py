import codecs
import json
import os
import random
import time
import numpy as np
import tensorflow as tf

from src import config
from src import util
from src.model import BiGRU


def pad_fact_batch(fact_batch):
    new_batch = []
    for fact in fact_batch:
        new_batch.append(util.pad_sequence(fact, config.SENTENCE_LEN, pad_type='id'))
    return new_batch


def inference(sess, model, batch_iter, out_file, verbose=True):
    task_1_output = []
    task_2_output = []
    task_3_output = []
    start_time = time.time()
    for i, batch in enumerate(batch_iter):
        if verbose:
            print('processing batch: %5d' % i, end='\r')

        fact, fact_len, _ = list(zip(*batch))

        batch_size = len(fact)
        fact = pad_fact_batch(fact)

        feed_dict = {
            model.fact: fact,
            model.fact_len: fact_len
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


def run_epoch(sess, model, batch_iter, verbose=True):
    steps = 0
    total_loss = 0.0
    _global_step = 0
    start_time = time.time()
    for batch in batch_iter:
        fact, fact_len, accu = list(zip(*batch))

        fact = pad_fact_batch(fact)

        feed_dict = {
            model.fact: fact,
            model.fact_len: fact_len,
            model.accu: accu
        }

        _, _loss, _global_step = sess.run(
            [model.train_op, model.loss, model.global_step],
            feed_dict=feed_dict
        )

        steps += 1
        total_loss += _loss
        if verbose and steps % 1000 == 1:
            current_time = time.time()
            print('After %5d batch(es), global step is %5d, loss is %.3f, cost time %.3fs'
                  % (steps, _global_step, _loss, current_time - start_time))
            start_time = current_time

    return total_loss / steps, _global_step


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


def read_data(data_file, word_2_id, accu_2_id, max_len):
    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    print('data size: ', len(lines))

    fact = []
    fact_len = []
    accu = []
    for line in lines:
        item = json.loads(line, encoding='utf-8')

        _fact = item['fact'].strip().lower()
        _fact = util.refine_text(_fact)
        _fact = util.convert_to_id_list(_fact, word_2_id)
        _fact = _fact[:max_len]
        fact.append(_fact)

        fact_len.append(len(_fact))

        temp = item['meta']['accusation']
        for i in range(len(temp)):
            temp[i] = temp[i].replace('[', '').replace(']', '')
        temp = [accu_2_id[v] for v in temp]
        _accu = [0] * config.ACCU_NUM
        for i in temp:
            _accu[i] = 1
        accu.append(_accu)

    return fact, fact_len, accu


def train(judger, config_proto):
    assert config.CURRENT_MODEL == 'bigru'

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
        train_model = BiGRU(
            accu_num=config.ACCU_NUM,
            max_seq_len=config.SENTENCE_LEN, hidden_size=config.HIDDEN_SIZE, fc_size=config.FC_SIZE_S,
            embedding_matrix=embedding_matrix, embedding_trainable=embedding_trainable,
            lr=config.LR, optimizer=config.OPTIMIZER, keep_prob=config.KEEP_PROB, l2_rate=config.L2_RATE,
            is_training=True
        )
    with tf.variable_scope('model', reuse=True):
        valid_model = BiGRU(
            accu_num=config.ACCU_NUM,
            max_seq_len=config.SENTENCE_LEN, hidden_size=config.HIDDEN_SIZE, fc_size=config.FC_SIZE_S,
            embedding_matrix=embedding_matrix, embedding_trainable=embedding_trainable,
            lr=config.LR, optimizer=config.OPTIMIZER, keep_prob=config.KEEP_PROB, l2_rate=config.L2_RATE,
            is_training=False
        )

    train_data = read_data(config.TRAIN_DATA, word_2_id, accu_2_id, max_len=config.SENTENCE_LEN)
    valid_data = read_data(config.VALID_DATA, word_2_id, accu_2_id, max_len=config.SENTENCE_LEN)

    best_accu_micro_f1 = 0.0
    best_accu_macro_f1 = 0.0
    best_article_micro_f1 = 0.0
    best_article_macro_f1 = 0.0
    best_score = [0.0, 0.0, 0.0]
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=config_proto) as sess:
        tf.global_variables_initializer().run()
        saver.save(sess, config.MODEL_FILE)

        for i in range(config.NUM_EPOCH):
            print('==========  Epoch %2d Train  ==========' % (i + 1))
            train_batch_iter = make_batch_iter(list(zip(*train_data)), config.BATCH_SIZE, shuffle=True)
            train_loss, global_step = run_epoch(sess, train_model, train_batch_iter, verbose=True)
            print('The average train loss of epoch %2d is %.3f' % ((i + 1), train_loss))

            print('==========  Epoch %2d Valid  ==========' % (i + 1))
            valid_batch_iter = make_batch_iter(list(zip(*valid_data)), config.BATCH_SIZE, shuffle=False)
            inference(sess, valid_model, valid_batch_iter, config.VALID_RESULT, verbose=True)

            # 单标签
            # result = judger.my_test(config.VALID_DATA, config.VALID_RESULT + '.json')
            # accu_micro_f1, accu_macro_f1 = judger.calc_f1(result[0])
            # article_micro_f1, article_macro_f1 = judger.calc_f1(result[1])
            # score = judger.get_score(result)
            # print('Threshold: %.3f' % threshold)
            # print('Micro-F1 of accusation: %.3f' % accu_micro_f1)
            # print('Macro-F1 of accusation: %.3f' % accu_macro_f1)
            # print('Micro-F1 of relevant articles: %.3f' % article_micro_f1)
            # print('Macro-F1 of relevant articles: %.3f' % article_macro_f1)
            # print('Score: ', score)
            #
            # if best_score[0] < score[0]:
            #     best_accu_micro_f1 = accu_micro_f1
            #     best_accu_macro_f1 = accu_macro_f1
            #     best_article_micro_f1 = article_micro_f1
            #     best_article_macro_f1 = article_macro_f1
            #     best_score = score
            #     print('Saving model ...')
            #     saver.save(sess, config.MODEL_FILE)

            # 多标签
            for threshold in config.TASK_THRESHOLD:
                result = judger.my_test(config.VALID_DATA, config.VALID_RESULT + '-' + str(threshold) + '.json')
                accu_micro_f1, accu_macro_f1 = judger.calc_f1(result[0])
                article_micro_f1, article_macro_f1 = judger.calc_f1(result[1])
                score = judger.get_score(result)
                print('Threshold: %.3f' % threshold)
                print('Micro-F1 of accusation: %.3f' % accu_micro_f1)
                print('Macro-F1 of accusation: %.3f' % accu_macro_f1)
                print('Micro-F1 of relevant articles: %.3f' % article_micro_f1)
                print('Macro-F1 of relevant articles: %.3f' % article_macro_f1)
                print('Score: ', score)

                if best_score[0] < score[0]:
                    best_accu_micro_f1 = accu_micro_f1
                    best_accu_macro_f1 = accu_macro_f1
                    best_article_micro_f1 = article_micro_f1
                    best_article_macro_f1 = article_macro_f1
                    best_score = score
                    print('Saving model ...')
                    saver.save(sess, config.MODEL_FILE)

    print('Best micro-F1 of accusation is: %.3f' % best_accu_micro_f1)
    print('Best macro-F1 of accusation is: %.3f' % best_accu_macro_f1)
    print('Best micro-F1 of relevant articles: %.3f' % best_article_micro_f1)
    print('Best macro-F1 of relevant articles: %.3f' % best_article_macro_f1)
    print('Best score is: ', best_score)
