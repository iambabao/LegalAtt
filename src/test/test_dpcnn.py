import codecs
import json
import os
import time
import numpy as np
import tensorflow as tf

from src.util import read_dict, init_dict, load_embedding, make_batch_iter, id_2_imprisonment, get_task_result
from src.data_reader import read_data, pad_fact_batch
from src.model import DPCNN


def inference(sess, model, batch_iter, config, verbose=True):
    task_1_output = []
    task_2_output = []
    task_3_output = []
    start_time = time.time()
    for i, batch in enumerate(batch_iter):
        if verbose:
            print('processing batch: %5d' % i, end='\r')

        fact, _, _, _, _ = list(zip(*batch))

        batch_size = len(fact)
        fact = pad_fact_batch(fact, config)

        feed_dict = {
            model.fact: fact
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

        print('write file: ', config.test_result + '-' + str(threshold) + '.json')
        with codecs.open(config.test_result + '-' + str(threshold) + '.json', 'w', encoding='utf-8') as f_out:
            for r in result:
                print(json.dumps(r), file=f_out)


def test(config, judger, config_proto):
    assert config.current_model == 'dpcnn'

    word_2_id, id_2_word = read_dict(config.word_dict)
    law_2_id, id_2_law, accu_2_id, id_2_accu = init_dict(config.law_dict, config.accu_dict)

    if os.path.exists(config.word2vec_model):
        embedding_matrix = load_embedding(config.word2vec_model, word_2_id.keys())
    else:
        embedding_matrix = np.random.uniform(-0.5, 0.5, [len(word_2_id), config.embedding_size])

    with tf.variable_scope('model', reuse=None):
        test_model = DPCNN(
            accu_num=config.accu_num, article_num=config.article_num, max_seq_len=config.sentence_len,
            kernel_size=config.kernel_size[1], filter_dim=config.filter_dim, fc_size=config.fc_size_s,
            embedding_matrix=embedding_matrix, embedding_trainable=config.embedding_trainable,
            lr=config.lr, optimizer=config.optimizer, keep_prob=config.keep_prob, l2_rate=config.l2_rate,
            use_batch_norm=config.use_batch_norm, is_training=False
        )

    test_data = read_data(config.test_data, word_2_id, accu_2_id, law_2_id, config)

    saver = tf.train.Saver()
    with tf.Session(config=config_proto) as sess:
        print('load model from: ' + config.model_file)
        saver.restore(sess, config.model_file)

        print('==========  Test  ==========')
        test_batch_iter = make_batch_iter(list(zip(*test_data)), config.batch_size, shuffle=False)
        inference(sess, test_model, test_batch_iter, config, verbose=True)

        for threshold in config.task_threshold:
            result = judger.test(config.test_data, config.test_result + '-' + str(threshold) + '.json')
            accu_micro_f1, accu_macro_f1, accu_res = judger.calc_f1(result[0])
            with codecs.open(config.accu_result + '-' + str(threshold) + '.json', 'w', encoding='utf-8') as fout:
                for i in range(config.accu_num):
                    accu_res[i]['accu'] = id_2_accu[i]
                    print(json.dumps(accu_res[i], ensure_ascii=False), file=fout)
            article_micro_f1, article_macro_f1, article_res = judger.calc_f1(result[1])
            with codecs.open(config.article_result + '-' + str(threshold) + '.json', 'w', encoding='utf-8') as fout:
                for i in range(config.article_num):
                    article_res[i]['art'] = id_2_law[i]
                    print(json.dumps(article_res[i], ensure_ascii=False), file=fout)
            score = [(accu_micro_f1 + accu_macro_f1) / 2, (article_micro_f1 + article_macro_f1) / 2]
            print('Threshold: %.3f' % threshold)
            print('Micro-F1 of accusation: %.3f' % accu_micro_f1)
            print('Macro-F1 of accusation: %.3f' % accu_macro_f1)
            print('Micro-F1 of relevant articles: %.3f' % article_micro_f1)
            print('Macro-F1 of relevant articles: %.3f' % article_macro_f1)
            print('Score: ', score)
