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
            print('processing batch: %5d' % i, end='\r')

        fact, fact_len, _, _, _ = list(zip(*batch))
        fact = pad_batch(fact, config.pad_id, config.sequence_len)
        bs = len(fact)

        feed_dict = {
            model.batch_size: bs,
            model.fact: fact,
            model.fact_len: fact_len,
            model.art: [art] * bs,
            model.art_len: [art_len] * bs
        }

        _task_1_output, _task_2_output = sess.run(
            [model.task_1_output, model.task_2_output],
            feed_dict=feed_dict
        )

        task_1_output.extend(_task_1_output.tolist())
        task_2_output.extend(_task_2_output.tolist())
        task_3_output.extend([[0.0] * config.impr_num] * bs)
    print('\ncost time: %.3fs' % (time.time() - start_time))

    return task_1_output, task_2_output, task_3_output


def test():
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    print('load data')
    word_2_id, id_2_word = read_dict(config.word_dict)
    accu_2_id, id_2_accu = read_dict(config.accu_dict)
    art_2_id, id_2_art = read_dict(config.art_dict)

    if os.path.exists(config.word2vec_model):
        embedding_matrix = load_embedding(config.word2vec_model, word_2_id.keys())
    else:
        embedding_matrix = np.random.uniform(-0.5, 0.5, [len(word_2_id), config.embedding_size])

    data_reader = DataReader(config)
    test_data = data_reader.read_test_data(word_2_id, accu_2_id, art_2_id)
    art_data = data_reader.read_article(art_2_id.keys(), word_2_id)

    print('build model')
    with tf.variable_scope('model'):
        model = get_model(config, embedding_matrix, is_training=False)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=config_proto) as sess:
        print('load model from: ' + config.model_file)
        saver.restore(sess, config.model_file)

        print('==========  Test  ==========')
        test_batch_iter = make_batch_iter(list(zip(*test_data)), config.batch_size, shuffle=False)
        outputs = inference(sess, model, test_batch_iter, art_data, verbose=True)

        save_result(outputs, config.test_result, id_2_accu, id_2_art)
        result = judger.get_result(config.test_data, config.test_result)
        accu_micro_f1, accu_macro_f1 = judger.calc_f1(result[0])
        article_micro_f1, article_macro_f1 = judger.calc_f1(result[1])
        score = [(accu_micro_f1 + accu_macro_f1) / 2, (article_micro_f1 + article_macro_f1) / 2]
        print('Micro-F1 of accusation: %.4f' % accu_micro_f1)
        print('Macro-F1 of accusation: %.4f' % accu_macro_f1)
        print('Micro-F1 of relevant articles: %.4f' % article_micro_f1)
        print('Macro-F1 of relevant articles: %.4f' % article_macro_f1)
        print('Score: ', score)


if __name__ == '__main__':
    test()
