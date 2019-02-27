import tensorflow as tf


class FactLaw(object):
    def __init__(self, accu_num, article_num,
                 top_k, tfidf_size, max_seq_len, max_doc_len,
                 hidden_size, att_size, fc_size,
                 embedding_matrix, embedding_trainable,
                 lr, optimizer, keep_prob, l2_rate, is_training):
        self.accu_num = accu_num
        self.article_num = article_num

        self.top_k = top_k
        self.tfidf_size = tfidf_size
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.hidden_size = hidden_size
        self.att_size = att_size
        self.fc_size = fc_size

        self.embedding_matrix = tf.get_variable(
            initializer=tf.constant_initializer(embedding_matrix),
            shape=embedding_matrix.shape,
            trainable=embedding_trainable,
            dtype=tf.float32,
            name='embedding_matrix')
        self.embedding_size = embedding_matrix.shape[-1]

        self.lr = lr
        self.optimizer = optimizer
        self.keep_prob = keep_prob
        self.l2_rate = l2_rate

        self.is_training = is_training

        self.w_init = tf.truncated_normal_initializer(stddev=0.1)
        self.b_init = tf.constant_initializer(0.1)

        if l2_rate > 0.0:
            self.regularizer = tf.contrib.layers.l2_regularizer(l2_rate)
        else:
            self.regularizer = None

        self.fact = tf.placeholder(dtype=tf.int32, shape=[None, max_doc_len, max_seq_len], name='fact')
        self.fact_seq_len = tf.placeholder(dtype=tf.int32, shape=[None, max_doc_len], name='fact_seq_len')
        self.fact_doc_len = tf.placeholder(dtype=tf.int32, shape=[None], name='fact_doc_len')
        self.tfidf = tf.placeholder(dtype=tf.float32, shape=[None, tfidf_size], name='tfidf')
        self.law_kb = tf.placeholder(dtype=tf.int32, shape=[None, article_num, max_doc_len, max_seq_len], name='law_kb')
        self.law_seq_len = tf.placeholder(dtype=tf.int32, shape=[None, article_num, max_doc_len], name='law_seq_len')
        self.law_doc_len = tf.placeholder(dtype=tf.int32, shape=[None, article_num], name='law_doc_len')
        self.accu = tf.placeholder(dtype=tf.float32, shape=[None, accu_num], name='accu')
        self.article = tf.placeholder(dtype=tf.float32, shape=[None, article_num], name='article')

        with tf.variable_scope('article_extractor'):
            # art_score's shape = [batch_size, article_num]
            # top_k_indices' shape = [batch_size, top_k]
            art_score, top_k_indices = self.get_top_k_indices()

            # top_k_art's shape = [batch_size, top_k, max_doc_len, max_seq_len]
            # top_k_art_seq_len's shape = [batch_size, top_k, max_doc_len]
            # top_k_art_doc_len's shape = [batch_size, top_k]
            top_k_art, top_k_art_seq_len, top_k_art_doc_len = self.get_top_k_articles(top_k_indices)

        with tf.variable_scope('fact_embedding'):
            fact_em = self.fact_embedding_layer()

        with tf.variable_scope('article_embedding'):
            top_k_art_em = self.article_embedding_layer(top_k_art)

        with tf.variable_scope('fact_encoder'):
            u_fw = tf.get_variable(initializer=self.w_init, shape=[self.att_size], name='u_fw')
            u_fs = tf.get_variable(initializer=self.w_init, shape=[self.att_size], name='u_fs')
            # fact_enc's shape = [batch_size, 2 * hidden_size]
            fact_enc = self.document_encoder(fact_em, self.fact_seq_len, self.fact_doc_len, u_fw, u_fs)

        with tf.variable_scope('article_encoder', reuse=tf.AUTO_REUSE):
            u_aw = tf.layers.dense(fact_enc, self.att_size, kernel_regularizer=self.regularizer)
            u_as = tf.layers.dense(fact_enc, self.att_size, kernel_regularizer=self.regularizer)
            art_em_splits = tf.split(top_k_art_em, self.top_k, axis=1)
            art_seq_len_splits = tf.split(top_k_art_seq_len, self.top_k, axis=1)
            art_doc_len_splits = tf.split(top_k_art_doc_len, self.top_k, axis=1)
            enc_outputs = []
            for art_em, seq_len, doc_len in zip(art_em_splits, art_seq_len_splits, art_doc_len_splits):
                enc_output = self.document_encoder(art_em, seq_len, doc_len, u_aw, u_as)
                enc_output = tf.expand_dims(enc_output, axis=1)
                enc_outputs.append(enc_output)
            # art_enc's shape = [batch_size, top_k, 2 * hidden_size]
            art_enc = tf.concat(enc_outputs, axis=1)

        with tf.variable_scope('article_aggregator'):
            u_ad = tf.layers.dense(fact_enc, self.att_size, kernel_regularizer=self.regularizer)
            # art_agg's shape = [batch_size, 2 * hidden_size]
            # art_att's shape = [batch_size, top_k, 1]
            agg_art_enc, art_att = self.article_aggregator(art_enc, u_ad)

        with tf.variable_scope('concat_layer'):
            final_output = tf.concat([fact_enc, agg_art_enc], axis=-1)

        with tf.variable_scope('output_layer'):
            self.task_1_output, task_1_loss = self.output_layer(final_output, self.accu, self.accu_num)

        with tf.variable_scope('loss'):
            task_2_loss = tf.losses.hinge_loss(labels=self.article, logits=art_score)

            self.loss = task_1_loss + task_2_loss
            if self.regularizer is not None:
                l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self.loss += l2_loss

        if not is_training:
            return

        self.global_step, self.train_op = self.get_train_op()

    def get_top_k_indices(self):
        scores = tf.layers.dense(
            self.tfidf,
            self.article_num,
            tf.nn.tanh,
            kernel_regularizer=self.regularizer
        )

        if self.is_training:
            _, indices = tf.math.top_k(self.article, k=self.top_k)
        else:
            _, indices = tf.math.top_k(scores, k=self.top_k)

        return scores, indices

    def get_top_k_articles(self, top_k_indices):
        art = tf.batch_gather(self.law_kb, indices=top_k_indices)
        art_seq_len = tf.batch_gather(self.law_seq_len, indices=top_k_indices)
        art_doc_len = tf.batch_gather(self.law_doc_len, indices=top_k_indices)

        return art, art_seq_len, art_doc_len

    def fact_embedding_layer(self):
        fact_em = tf.nn.embedding_lookup(self.embedding_matrix, self.fact)
        if self.is_training and self.keep_prob < 1.0:
            fact_em = tf.nn.dropout(fact_em, keep_prob=self.keep_prob)

        return fact_em

    def article_embedding_layer(self, art):
        art_em = tf.nn.embedding_lookup(self.embedding_matrix, art)
        if self.is_training and self.keep_prob < 1.0:
            art_em = tf.nn.dropout(art_em, keep_prob=self.keep_prob)

        return art_em

    def document_encoder(self, inputs, seq_len, doc_len, u_w, u_s):
        with tf.variable_scope('sequence_level'):
            inputs = tf.reshape(inputs, [-1, self.max_seq_len, self.embedding_size])
            seq_len = tf.reshape(seq_len, [-1])

            cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs,
                sequence_length=seq_len,
                dtype=tf.float32
            )
            # seq_output's shape = [batch_size, doc_len, seq_len, 2 * hidden_size]
            seq_output = tf.concat([output_fw, output_bw], axis=-1)
            seq_output = tf.reshape(seq_output, [-1, self.max_doc_len, self.max_seq_len, 2 * self.hidden_size])

            # att_w's shape = [batch_size, doc_len, seq_len, 1]
            u = tf.layers.dense(seq_output, self.att_size, tf.nn.tanh, kernel_regularizer=self.regularizer)
            u_att = tf.reshape(u_w, [-1, 1, 1, self.att_size])
            att_w = tf.math.softmax(tf.reduce_sum(u * u_att, axis=-1, keepdims=True), axis=-1)

            # seq_output's shape = [batch_size, doc_len, 2 * hidden_size]
            seq_output = tf.reduce_sum(att_w * seq_output, axis=-2)

        with tf.variable_scope('document_level'):
            doc_len = tf.reshape(doc_len, [-1])

            cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=seq_output,
                sequence_length=doc_len,
                dtype=tf.float32
            )
            # doc_output's shape = [batch_size, doc_len, 2 * hidden_size]
            doc_output = tf.concat([output_fw, output_bw], axis=-1)

            # att_s' shape = [batch_size, doc_len, 1]
            u = tf.layers.dense(doc_output, self.att_size, tf.nn.tanh, kernel_regularizer=self.regularizer)
            u_att = tf.reshape(u_s, [-1, 1, self.att_size])
            att_s = tf.math.softmax(tf.reduce_sum(u * u_att, axis=-1, keepdims=True), axis=-1)

            # doc_output's shape = [batch_size, 2 * hidden_size]
            doc_output = tf.reduce_sum(att_s * doc_output, axis=-2)

        return doc_output

    def article_aggregator(self, art, u_document):
        gru_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        gru_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)

        # outputs' shape = [batch_size, top_k, 2 * hidden_size]
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=gru_fw,
            cell_bw=gru_bw,
            inputs=art,
            dtype=tf.float32
        )
        output = tf.concat([output_fw, output_bw], axis=-1)

        # att's shape = [batch_size, top_k, 1]
        u = tf.layers.dense(output, self.att_size, tf.nn.tanh, kernel_regularizer=self.regularizer)
        u_att = tf.reshape(u_document, [-1, 1, self.att_size])
        att_d = tf.math.softmax(tf.reduce_sum(u * u_att, axis=-1, keepdims=True), axis=-1)

        # art_outputs' shape = [batch_size, 2 * hidden_size]
        art_outputs = tf.reduce_sum(output * att_d, axis=-2)

        return art_outputs, att_d

    def output_layer(self, inputs, labels, label_num):
        fc_output = tf.layers.dense(inputs, self.fc_size, kernel_regularizer=self.regularizer)
        if self.is_training and self.keep_prob < 1.0:
            fc_output = tf.nn.dropout(fc_output, keep_prob=self.keep_prob)

        logits = tf.layers.dense(fc_output, label_num, kernel_regularizer=self.regularizer)
        output = tf.nn.sigmoid(logits)

        ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

        return output, ce_loss

    def get_train_op(self):
        global_step = tf.Variable(0, trainable=False, name='global_step')

        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.optimizer == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
        elif self.optimizer == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        train_op = optimizer.minimize(self.loss, global_step=global_step)
        return global_step, train_op
