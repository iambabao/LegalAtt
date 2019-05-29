import os
import time
import json
import argparse
import numpy as np
import tensorflow as tf
import jieba

from src.config import Config
from src.judger import Judger
from src.data_reader import DataReader
from src.model import get_model
from src.util import read_dict, load_embedding, make_batch_iter, pad_batch, get_task_result, id_2_impr

jieba.add_word('PAD', 9999, 'n')
jieba.add_word('UNK', 9999, 'n')
jieba.add_word('NUM', 9999, 'n')
jieba.add_word('TIME', 9999, 'n')

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--embedding_trainable', action='store_true', default=False)
parser.add_argument('--use_batch_norm', action='store_true', default=False)
args = parser.parse_args()

current_model = args.model
num_epoch = args.num_epoch
batch_size = args.batch_size
optimizer = args.optimizer
lr = args.lr
embedding_trainable = args.embedding_trainable
use_batch_norm = args.use_batch_norm
config = Config('./', current_model,
                num_epoch=num_epoch, batch_size=batch_size, optimizer=optimizer, lr=lr,
                embedding_trainable=embedding_trainable, use_batch_norm=use_batch_norm)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_proto = tf.ConfigProto(allow_soft_placement=True)  # 创建配置，允许将无法放入GPU的操作放在CUP上执行
config_proto.gpu_options.allow_growth = True  # 运行时动态增加内存使用量
judger = Judger(config.accu_dict, config.art_dict)


def save_result(outputs, result_file, id_2_accu, id_2_art):
    task_1_output, task_2_output, task_3_output = outputs

    task_1_result = [get_task_result(s, config.threshold) for s in task_1_output]
    task_2_result = [get_task_result(s, config.threshold) for s in task_2_output]
    task_3_result = np.argmax(task_3_output, axis=-1)

    print('write file: ', result_file)
    with open(result_file, 'w', encoding='utf-8') as fout:
        for t1, t2, t3 in zip(task_1_result, task_2_result, task_3_result):
            t1 = [id_2_accu[v] for v in t1]
            t2 = [int(id_2_art[v]) for v in t2]
            t3 = id_2_impr(t3)

            res = {
                'accusation': t1,
                'relevant_articles': t2,
                'imprisonment': t3
            }
            print(json.dumps(res, ensure_ascii=False), file=fout)


def inference(sess, model, batch_iter, art_data, verbose=True):
    art, art_len = art_data

    task_1_output = []
    task_2_output = []
    task_3_output = []
    start_time = time.time()
    for i, batch in enumerate(batch_iter):
        if verbose:
            print('processing batch: %6d' % i, end='\r')

        fact, fact_len, accu, relevant_art, impr = list(zip(*batch))
        fact = pad_batch(fact, config.pad_id, config.sequence_len)
        bs = len(fact)

        feed_dict = {
            model.batch_size: bs,
            model.fact: fact,
            model.fact_len: fact_len,
            model.art: [art] * bs,
            model.art_len: [art_len] * bs,
            model.accu: accu,
            model.relevant_art: relevant_art,
            model.impr: impr
        }

        _task_1_output, _task_2_output = sess.run(
            [model.task_1_output, model.task_2_output],
            feed_dict=feed_dict
        )

        task_1_output.extend(_task_1_output.tolist())
        task_2_output.extend(_task_2_output.tolist())
        task_3_output.extend([[0.0] * config.impr_num] * bs)
    print('\ncost time: %.4fs' % (time.time() - start_time))

    return task_1_output, task_2_output, task_3_output


def run_epoch(sess, model, batch_iter, art_data, verbose=True):
    art, art_len = art_data

    steps = 0
    total_loss = 0.0
    _global_step = 0
    start_time = time.time()
    for batch in batch_iter:
        fact, fact_len, accu, relevant_art, impr = list(zip(*batch))
        fact = pad_batch(fact, config.pad_id, config.sequence_len)
        bs = len(fact)

        feed_dict = {
            model.batch_size: bs,
            model.fact: fact,
            model.fact_len: fact_len,
            model.art: [art] * bs,
            model.art_len: [art_len] * bs,
            model.accu: accu,
            model.relevant_art: relevant_art,
            model.impr: impr
        }

        _, _loss, _global_step = sess.run(
            [model.train_op, model.loss, model.global_step],
            feed_dict=feed_dict
        )

        steps += 1
        total_loss += _loss
        if verbose and steps % 1000 == 1:
            current_time = time.time()
            print('After %6d batch(es), global step is %6d, loss is %.4f, cost time %.4fs'
                  % (steps, _global_step, _loss, current_time - start_time))
            start_time = current_time

    return total_loss / steps, _global_step


def train():
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    print('load data...')
    word_2_id, id_2_word = read_dict(config.word_dict)
    accu_2_id, id_2_accu = read_dict(config.accu_dict)
    art_2_id, id_2_art = read_dict(config.art_dict)

    if os.path.exists(config.word2vec_model):
        embedding_matrix = load_embedding(config.word2vec_model, word_2_id.keys())
    else:
        embedding_matrix = np.random.uniform(-0.5, 0.5, [len(word_2_id), config.embedding_size])

    data_reader = DataReader(config)
    train_data = data_reader.read_train_data(word_2_id, accu_2_id, art_2_id)
    valid_data = data_reader.read_valid_data(word_2_id, accu_2_id, art_2_id)
    art_data = data_reader.read_article(art_2_id.keys(), word_2_id)

    print('build model...')
    with tf.variable_scope('model'):
        model = get_model(config, embedding_matrix, is_training=True)

    print('==========  Trainable Variables  ==========')
    for v in tf.trainable_variables():
        print(v)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=config_proto) as sess:
        tf.global_variables_initializer().run()
        saver.save(sess, config.model_file)

        for i in range(config.num_epoch):
            print('==========  Epoch %2d Train  ==========' % (i + 1))
            train_batch_iter = make_batch_iter(list(zip(*train_data)), config.batch_size, shuffle=True)
            train_loss, _ = run_epoch(sess, model, train_batch_iter, art_data, verbose=True)
            print('The average train loss of epoch %2d is %.4f' % ((i + 1), train_loss))

            print('==========  Epoch %2d Valid  ==========' % (i + 1))
            valid_batch_iter = make_batch_iter(list(zip(*valid_data)), config.batch_size, shuffle=False)
            outputs = inference(sess, model, valid_batch_iter, art_data, verbose=True)

            print('==========  Saving model  ==========')
            saver.save(sess, config.model_file)

            save_result(outputs, config.valid_result, id_2_accu, id_2_art)
            result = judger.get_result(config.valid_data, config.valid_result)
            accu_micro_f1, accu_macro_f1 = judger.calc_f1(result[0])
            article_micro_f1, article_macro_f1 = judger.calc_f1(result[1])
            score = [(accu_micro_f1 + accu_macro_f1) / 2, (article_micro_f1 + article_macro_f1) / 2]
            print('Micro-F1 of accusation: %.4f' % accu_micro_f1)
            print('Macro-F1 of accusation: %.4f' % accu_macro_f1)
            print('Micro-F1 of relevant articles: %.4f' % article_micro_f1)
            print('Macro-F1 of relevant articles: %.4f' % article_macro_f1)
            print('Score: ', score)


if __name__ == '__main__':
    train()
