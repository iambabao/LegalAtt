import codecs
import json
import os
import time
import numpy as np
import tensorflow as tf

from src.util import read_dict, init_dict, load_embedding, make_batch_iter, id_2_imprisonment, get_task_result
from src.data_reader import read_data_doc, pad_fact_batch_doc
from src.model import HAN


def inference(sess, model, batch_iter, config, verbose=True):
    task_1_output = []
    task_2_output = []
    task_3_output = []
    start_time = time.time()
    for i, batch in enumerate(batch_iter):
        if verbose:
            print('processing batch: %5d' % i, end='\r')

        fact, fact_seq_len, fact_doc_len, accu, _, _ = list(zip(*batch))

        batch_size = len(fact)
        fact = pad_fact_batch_doc(fact, config)

        feed_dict = {
            model.fact: fact,
            model.fact_seq_len: fact_seq_len,
            model.fact_doc_len: fact_doc_len
        }

        _task_1_output = sess.run(
            model.task_1_output,
            feed_dict=feed_dict
        )
        task_1_output.extend(_task_1_output.tolist())
        task_2_output.extend([[0.0] * config.article_num] * batch_size)
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
    assert config.current_model == 'han'

    word_2_id, id_2_word = read_dict(config.word_dict)
    law_2_id, id_2_law, accu_2_id, id_2_accu = init_dict(config.law_dict, config.accu_dict)

    if os.path.exists(config.word2vec_model):
        embedding_matrix = load_embedding(config.word2vec_model, word_2_id.keys())
    else:
        embedding_matrix = np.random.uniform(-0.5, 0.5, [len(word_2_id), config.embedding_size])

    with tf.variable_scope('model', reuse=None):
        test_model = HAN(
            accu_num=config.accu_num, max_seq_len=config.sequence_len, max_doc_len=config.document_len,
            hidden_size=config.hidden_size, att_size=config.att_size, fc_size=config.fc_size_s,
            embedding_matrix=embedding_matrix, embedding_trainable=config.embedding_trainable,
            lr=config.lr, optimizer=config.optimizer, keep_prob=config.keep_prob, l2_rate=config.l2_rate,
            use_batch_norm=config.use_batch_norm, is_training=False
        )

    test_data = read_data_doc(config.test_data, word_2_id, accu_2_id, law_2_id, config)

    saver = tf.train.Saver()
    with tf.Session(config=config_proto) as sess:
        print('load model from: ' + config.model_file)
        saver.restore(sess, config.model_file)

        print('==========  Test  ==========')
        test_batch_iter = make_batch_iter(list(zip(*test_data)), config.batch_size, shuffle=False)
        inference(sess, test_model, test_batch_iter, config, verbose=True)

        for threshold in config.task_threshold:
            result = judger.my_test(config.test_data, config.test_result + '-' + str(threshold) + '.json')
            accu_micro_f1, accu_macro_f1 = judger.calc_f1(result[0])
            article_micro_f1, article_macro_f1 = judger.calc_f1(result[1])
            score = judger.get_score(result)
            print('Threshold: %.3f' % threshold)
            print('Micro-F1 of accusation: %.3f' % accu_micro_f1)
            print('Macro-F1 of accusation: %.3f' % accu_macro_f1)
            print('Micro-F1 of relevant articles: %.3f' % article_micro_f1)
            print('Macro-F1 of relevant articles: %.3f' % article_macro_f1)
            print('Score: ', score)
