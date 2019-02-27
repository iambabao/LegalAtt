import tensorflow as tf


class LawAtt(object):
    def __init__(self, accu_num, article_num,
                 top_k, max_seq_len, hidden_size, att_size, fc_size,
                 embedding_matrix, embedding_trainable,
                 lr, optimizer, keep_prob, l2_rate, is_training):
        self.accu_num = accu_num
        self.article_num = article_num

        self.top_k = top_k
        self.max_seq_len = max_seq_len
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

        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[1], name='batch_size')
        self.fact = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len], name='fact')
        self.fact_len = tf.placeholder(dtype=tf.int32, shape=[None], name='fact_len')
        self.law_kb = tf.placeholder(dtype=tf.int32, shape=[None, article_num, max_seq_len], name='law_kb')
        self.law_len = tf.placeholder(dtype=tf.int32, shape=[None, article_num], name='law_len')
        self.accu = tf.placeholder(dtype=tf.float32, shape=[None, accu_num], name='accu')
        self.article = tf.placeholder(dtype=tf.float32, shape=[None, article_num], name='article')

        with tf.variable_scope('fact_embedding'):
            fact_em = self.fact_embedding_layer()

        with tf.variable_scope('fact_encoder'):
            # fact_enc's shape = [batch_size, max_seq_len, 2 * hidden_size]
            fact_enc = self.fact_encoder(fact_em, self.fact_len)

        with tf.variable_scope('article_extractor'):
            # art_score's shape = [batch_size, article_num]
            # top_k_indices' shape = [batch_size, top_k]
            art_score, top_k_indices = self.get_top_k_indices(fact_enc)

            # art's shape = [batch_size, top_k, max_seq_len]
            # art_len's shape = [batch_size, top_k]
            top_k_art, top_k_art_len = self.get_top_k_articles(top_k_indices)

        with tf.variable_scope('article_embedding'):
            top_k_art_em = self.article_embedding_layer(top_k_art)

        # set reuse to tf.AUTO_REUSE to allow all articles use the same gru
        with tf.variable_scope('article_encoder', reuse=tf.AUTO_REUSE):
            art_em_splits = tf.split(top_k_art_em, self.top_k, axis=1)
            art_len_splits = tf.split(top_k_art_len, self.top_k, axis=1)

            # top_k_art_enc's shape = [batch_size, 2 * hidden_size] * top_k
            top_k_art_enc = []
            for art_em, art_len in zip(art_em_splits, art_len_splits):
                enc_output = self.article_encoder(art_em, art_len)
                top_k_art_enc.append(enc_output)

        with tf.variable_scope('attention_layer'):
            key = tf.layers.dense(
                fact_enc,
                self.att_size,
                tf.nn.tanh,
                use_bias=False,
                kernel_regularizer=self.regularizer
            )

            w = tf.get_variable(
                initializer=self.w_init,
                shape=[2 * self.hidden_size, self.att_size],
                dtype=tf.float32,
                name='w'
            )
            # att_matrix's shape = [batch_size, top_k, max_seq_len]
            att_matrix = []
            for art_enc in top_k_art_enc:
                query = tf.nn.tanh(tf.matmul(art_enc, w))
                att = self.get_attention(key, query)
                att = tf.expand_dims(att, axis=1)
                att_matrix.append(att)
            att_matrix = tf.concat(att_matrix, axis=1)

            # fact_enc's shape = [batch_size, 2 * hidden_size]
            fact_enc = tf.matmul(att_matrix, fact_enc)
            fact_enc = tf.reduce_max(fact_enc, axis=1)

        with tf.variable_scope('output_layer'):
            self.task_1_output, task_1_loss = self.output_layer(fact_enc, self.accu, self.accu_num)

        with tf.variable_scope('loss'):
            art_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.article, logits=art_score))

            # penalty_matrix = tf.matmul(att_matrix, att_matrix, transpose_b=True)
            # penalty_matrix = penalty_matrix - tf.linalg.eye(self.top_k, batch_shape=self.batch_size)
            # fro_norm = tf.linalg.norm(penalty_matrix, ord='fro', axis=[1, 2])
            # att_loss = tf.reduce_mean(fro_norm * fro_norm)

            self.loss = task_1_loss + art_loss
            if self.regularizer is not None:
                l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self.loss += l2_loss

        if not is_training:
            return

        self.global_step, self.train_op = self.get_train_op()

    def fact_embedding_layer(self):
        fact_em = tf.nn.embedding_lookup(self.embedding_matrix, self.fact)
        if self.is_training and self.keep_prob < 1.0:
            fact_em = tf.nn.dropout(fact_em, keep_prob=self.keep_prob)

        return fact_em

    def fact_encoder(self, inputs, seq_len):
        cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=seq_len,
            dtype=tf.float32
        )

        output = tf.concat([output_fw, output_bw], axis=-1)

        return output

    def get_top_k_indices(self, inputs):
        inputs = tf.reduce_sum(inputs, axis=-2)
        scores = tf.layers.dense(inputs, self.article_num, tf.nn.tanh, kernel_regularizer=self.regularizer)

        if self.is_training:
            _, indices = tf.math.top_k(self.article, k=self.top_k)
        else:
            _, indices = tf.math.top_k(scores, k=self.top_k)

        return scores, indices

    def get_top_k_articles(self, top_k_indices):
        art = tf.batch_gather(self.law_kb, indices=top_k_indices)
        art_len = tf.batch_gather(self.law_len, indices=top_k_indices)

        return art, art_len

    def article_embedding_layer(self, article):
        article_em = tf.nn.embedding_lookup(self.embedding_matrix, article)
        if self.is_training and self.keep_prob < 1.0:
            article_em = tf.nn.dropout(article_em, keep_prob=self.keep_prob)

        return article_em

    def article_encoder(self, inputs, seq_len):
        inputs = tf.reshape(inputs, [-1, self.max_seq_len, self.embedding_size])
        seq_len = tf.reshape(seq_len, [-1])

        cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=seq_len,
            dtype=tf.float32
        )

        output = tf.concat([state_fw, state_bw], axis=-1)

        return output

    def get_attention(self, key, query):
        query = tf.reshape(query, [-1, 1, self.att_size])
        att = tf.math.softmax(tf.reduce_sum(key * query, axis=-1), axis=-1)

        return att

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
