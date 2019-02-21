import codecs
import json
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib

import config
import util
from topjudge import FactHealth

os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_ID
configDevice = tf.ConfigProto(allow_soft_placement=True)  # 创建配置，允许将无法放入GPU的操作放在CUP上执行
configDevice.gpu_options.allow_growth = True  # 运行时动态增加内存使用量


def pad_text(batch, max_doc_len=config.MAX_DOC_LEN, max_seq_len=config.MAX_SEQ_LEN):
    for i, text in enumerate(batch):
        batch[i] = [util.pad_sequence(s, max_len=max_seq_len, pad_type='id') for s in text]
        batch[i] += [[config.PAD_ID] * max_seq_len] * (max_doc_len - len(text))

    return batch


def inference(sess, model, data, batch_size, id_2_disease, out_file, verbose=True):
    data_size = len(data)
    num_batches = (data_size + batch_size - 1) // batch_size
    fact, fact_seq_len, fact_doc_len, tfidf = (list(item) for item in zip(*data))

    fact_softmax = []
    art_softmax = []
    start_time = time.time()
    for i in range(num_batches):
        if verbose:
            print('processing: %5d / %5d' % (i + 1, num_batches), end='\r')

        start_index = i * batch_size
        end_index = min(data_size, (i + 1) * batch_size)
        fact_batch = fact[start_index:end_index]
        fact_seq_len_batch = fact_seq_len[start_index:end_index]
        fact_doc_len_batch = fact_doc_len[start_index:end_index]
        tfidf_batch = tfidf[start_index:end_index]

        fact_batch = pad_text(fact_batch)

        feed_dict = {
            model.fact: fact_batch,
            model.fact_seq_len: fact_seq_len_batch,
            model.fact_doc_len: fact_doc_len_batch,
            model.tfidf: tfidf_batch
        }

        _fact_softmax, _art_softmax = sess.run(
            [model.fact_softmax, model.art_softmax],
            feed_dict=feed_dict
        )

        fact_softmax.extend(_fact_softmax)
        art_softmax.extend(_art_softmax)

    print('\ncost time: %.3fs' % (time.time() - start_time))

    # 多标签预测，根据不同的阈值得到不同的结果
    for threshold in config.TASK_THRESHOLD:
        fact_result = [util.get_task_result(s, threshold) for s in fact_softmax]
        art_result = [util.get_task_result(s, threshold) for s in art_softmax]

        result = []
        for f, a in zip(fact_result, art_result):
            result.append({
                'labels': [id_2_disease[t] for t in f],
                'articles': [id_2_disease[t] for t in a]
            })

        print('write file: ', out_file + '-' + str(threshold))
        with codecs.open(out_file + '-' + str(threshold), 'w', encoding='utf-8') as f_out:
            for r in result:
                print(json.dumps(r, ensure_ascii=False), file=f_out)

    # 单标签预测
    # fact_result = [[np.argmax(s, axis=-1)] for s in fact_softmax]
    # art_result = [[np.argmax(s, axis=-1)] for s in art_softmax]
    #
    # result = []
    # for f, a in zip(fact_result, art_result):
    #     result.append({
    #         'labels': [id_2_disease[t] for t in f],
    #         'articles': [id_2_disease[t] for t in a]
    #     })
    #
    # print('write file: ', out_file)
    # with codecs.open(out_file, 'w', encoding='utf-8') as f_out:
    #     for r in result:
    #         print(json.dumps(r, ensure_ascii=False), file=f_out)


def read_test_data(data_file, word_2_id, max_doc_len=config.MAX_DOC_LEN, max_seq_len=config.MAX_SEQ_LEN):
    print('read file: ', data_file)
    with codecs.open(data_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    print('data size: ', len(lines))

    tfidf_model = joblib.load(config.TFIDF_MODEL_FILE)

    fact = []
    fact_seq_len = []
    fact_doc_len = []
    corpus = []
    for line in lines:
        item = json.loads(line, encoding='utf-8')

        _fact = item['feature'].split('。')
        _fact = [util.convert_to_id_list(s[:max_seq_len], word_2_id) for s in _fact]
        _fact = _fact[:max_doc_len]
        fact.append(_fact)
        fact_seq_len.append([len(s) for s in _fact] + [0] * (max_doc_len - len(_fact)))
        fact_doc_len.append(len(_fact))

        corpus.append(item['feature'])

    return fact, fact_seq_len, fact_doc_len, tfidf_model.transform(corpus).toarray()


def read_kb(kb_dir, id_2_disease, word_2_id, max_doc_len=config.MAX_DOC_LEN, max_seq_len=config.MAX_SEQ_LEN):
    kb = []
    kb_seq_len = []
    kb_doc_len = []
    for i in range(len(id_2_disease)):
        disease = id_2_disease[i]
        file = os.path.join(kb_dir, disease + '.txt')
        with codecs.open(file, 'r', encoding='utf-8') as f_in:
            text = f_in.readlines()
        text = [l.strip() for l in text]
        text = ''.join(text)
        text = text.replace(' ', '').replace('\r', '').replace('\n', '').lower()
        if len(text) == 0:
            text = file  # 如果文本为空，则填充疾病名称
        text = text.split('。')
        text = [util.convert_to_id_list(s[:max_seq_len], word_2_id) for s in text]
        text = text[:max_doc_len]

        kb.append(text)
        kb_seq_len.append([len(s) for s in text] + [0] * (max_doc_len - len(text)))
        kb_doc_len.append(len(text))

    return pad_text(kb), kb_seq_len, kb_doc_len


def predict():
    word_2_id, id_2_word = util.read_dict(config.WORD_DICT)
    vocab_size = min(config.VOCAB_SIZE, len(word_2_id))
    disease_2_id, id_2_disease = util.init_dict(config.DISEASE_DICT)
    kb, kb_seq_len, kb_doc_len = read_kb(config.KB_DIR, id_2_disease, word_2_id)

    if os.path.exists(config.EMBEDDING_FILE):
        word_embed = util.load_embedding(config.EMBEDDING_FILE, word_2_id.keys())
    else:
        word_embed = np.random.uniform(-0.5, 0.5, [vocab_size, config.EMBEDDING_SIZE])

    with tf.variable_scope('model', reuse=None):
        test_model = FactHealth(
            disease_num=config.DISEASE_NUM, max_doc_len=config.MAX_DOC_LEN, max_seq_len=config.MAX_SEQ_LEN,
            k_size=config.K_SIZE, kb=kb, kb_doc_len=kb_doc_len, kb_seq_len=kb_seq_len,
            tfidf_size=config.TFIDF_SIZE, word_embed=word_embed,
            hidden_size=config.HIDDEN_SIZE, att_size=config.ATT_SIZE, optimizer=config.OPTIMIZER,
            lr_base=config.LR_BASE, lr_decay_rate=config.LR_DECAY_RATE, lr_decay_step=config.LR_DECAY_STEP,
            keep_prob=config.KEEP_PROB, grad_clip=config.GRAD_CLIP, l2_rate=config.L2_RATE, is_training=False
        )

    test_data = read_test_data(config.TEST_DATA_FILE, word_2_id)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=configDevice) as sess:
        print('load model from: ', config.MODEL_FILE)
        saver.restore(sess, config.MODEL_FILE)

        inference(sess, test_model, list(zip(*test_data)), config.BATCH_SIZE, id_2_disease, config.TEST_RESULT_FILE)


if __name__ == '__main__':
    predict()
