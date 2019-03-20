import codecs
import json
import os
import time
import numpy as np
import tensorflow as tf

from src.util import read_dict, init_dict, load_embedding, make_batch_iter, id_2_imprisonment, get_task_result
from src.data_reader import read_data, pad_fact_batch, read_law_kb
from src.model import LawAtt


def inference(sess, model, batch_iter, kb_data, config, verbose=True):
    law_kb, law_len = kb_data

    task_1_output = []
    task_2_output = []
    task_3_output = []
    start_time = time.time()
    for i, batch in enumerate(batch_iter):
        if verbose:
            print('processing batch: %5d' % i, end='\r')

        fact, fact_len, _, _, _ = list(zip(*batch))

        batch_size = len(fact)
        fact = pad_fact_batch(fact, config)

        feed_dict = {
            model.fact: fact,
            model.fact_len: fact_len,
            model.law_kb: [law_kb] * batch_size,
            model.law_len: [law_len] * batch_size
        }

        _task_1_output, _task_2_output = sess.run(
            [model.task_1_output, model.task_2_output],
            feed_dict=feed_dict
        )

        task_1_output.extend(_task_1_output.tolist())
        task_2_output.extend(_task_2_output.tolist())
        task_3_output.extend([[0.0] * config.imprisonment_num] * batch_size)
    print('\ncost time: %.3fs' % (time.time() - start_time))

    for threshold in config.task_threshold:
        task_1_result = [get_task_result(s, threshold) for s in task_1_output]
        task_2_result = [get_task_result(s, threshold) for s in task_2_output]
        task_3_result = np.argmax(task_3_output, axis=-1)

        result = []
        for t1, t2, t3 in zip(task_1_result, task_2_result, task_3_result):
            result.append({
                'accusation': t1,
                'articles': t2,
                'imprisonment': id_2_imprisonment(t3),
            })

        print('write file: ', config.valid_result + '-' + str(threshold) + '.json')
        with codecs.open(config.valid_result + '-' + str(threshold) + '.json', 'w', encoding='utf-8') as f_out:
            for r in result:
                print(json.dumps(r), file=f_out)


def run_epoch(sess, model, batch_iter, kb_data, config, verbose=True):
    law_kb, law_len = kb_data

    steps = 0
    total_loss = 0.0
    _global_step = 0
    start_time = time.time()
    for batch in batch_iter:
        fact, fact_len, accu, article, _ = list(zip(*batch))

        batch_size = len(fact)
        fact = pad_fact_batch(fact, config)

        feed_dict = {
            model.fact: fact,
            model.fact_len: fact_len,
            model.law_kb: [law_kb] * batch_size,
            model.law_len: [law_len] * batch_size,
            model.accu: accu,
            model.article: article
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


def train(config, judger, config_proto):
    assert config.current_model == 'law_att'

    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    word_2_id, id_2_word = read_dict(config.word_dict)
    law_2_id, id_2_law, accu_2_id, id_2_accu = init_dict(config.law_dict, config.accu_dict)

    if os.path.exists(config.word2vec_model):
        embedding_matrix = load_embedding(config.word2vec_model, word_2_id.keys())
        embedding_trainable = False
    else:
        embedding_matrix = np.random.uniform(-0.5, 0.5, [len(word_2_id), config.embedding_size])
        embedding_trainable = True

    with tf.variable_scope('model', reuse=None):
        train_model = LawAtt(
            accu_num=config.accu_num, article_num=config.article_num,
            top_k=config.top_k, max_seq_len=config.sentence_len,
            hidden_size=config.hidden_size, att_size=config.att_size,
            kernel_size=config.kernel_size, filter_dim=config.filter_dim, fc_size=config.fc_size_s,
            embedding_matrix=embedding_matrix, embedding_trainable=embedding_trainable,
            lr=config.lr, optimizer=config.optimizer, keep_prob=config.keep_prob, l2_rate=config.l2_rate,
            is_training=True
        )
    with tf.variable_scope('model', reuse=True):
        valid_model = LawAtt(
            accu_num=config.accu_num, article_num=config.article_num,
            top_k=config.top_k, max_seq_len=config.sentence_len,
            hidden_size=config.hidden_size, att_size=config.att_size,
            kernel_size=config.kernel_size, filter_dim=config.filter_dim, fc_size=config.fc_size_s,
            embedding_matrix=embedding_matrix, embedding_trainable=embedding_trainable,
            lr=config.lr, optimizer=config.optimizer, keep_prob=config.keep_prob, l2_rate=config.l2_rate,
            is_training=False
        )

    train_data = read_data(config.train_data, word_2_id, accu_2_id, law_2_id, config)
    valid_data = read_data(config.valid_data, word_2_id, accu_2_id, law_2_id, config)
    kb_data = read_law_kb(id_2_law, word_2_id, config)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=config_proto) as sess:
        tf.global_variables_initializer().run()
        saver.save(sess, config.model_file)

        for i in range(config.num_epoch):
            print('==========  Epoch %2d Train  ==========' % (i + 1))
            train_batch_iter = make_batch_iter(list(zip(*train_data)), config.batch_size, shuffle=True)
            train_loss, global_step = run_epoch(sess, train_model, train_batch_iter, kb_data, config, verbose=True)
            print('The average train loss of epoch %2d is %.3f' % ((i + 1), train_loss))

            print('==========  Epoch %2d Valid  ==========' % (i + 1))
            valid_batch_iter = make_batch_iter(list(zip(*valid_data)), config.batch_size, shuffle=False)
            inference(sess, valid_model, valid_batch_iter, kb_data, config, verbose=True)

            print('==========  Saving model  ==========')
            saver.save(sess, config.model_file)
            for threshold in config.task_threshold:
                result = judger.my_test(config.valid_data, config.valid_result + '-' + str(threshold) + '.json')
                accu_micro_f1, accu_macro_f1 = judger.calc_f1(result[0])
                article_micro_f1, article_macro_f1 = judger.calc_f1(result[1])
                score = judger.get_score(result)
                print('Threshold: %.3f' % threshold)
                print('Micro-F1 of accusation: %.3f' % accu_micro_f1)
                print('Macro-F1 of accusation: %.3f' % accu_macro_f1)
                print('Micro-F1 of relevant articles: %.3f' % article_micro_f1)
                print('Macro-F1 of relevant articles: %.3f' % article_macro_f1)
                print('Score: ', score)
