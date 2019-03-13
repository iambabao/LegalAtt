import tensorflow as tf


class LawAtt(object):
    def __init__(self, accu_num, article_num, top_k, max_seq_len,
                 hidden_size, att_size, kernel_size, filter_dim, fc_size,
                 embedding_matrix, embedding_trainable,
                 lr, optimizer, keep_prob, l2_rate, is_training):
        self.accu_num = accu_num
        self.article_num = article_num
        self.top_k = top_k
        self.max_seq_len = max_seq_len

        self.hidden_size = hidden_size
        self.att_size = att_size
        self.kernel_size = kernel_size
        self.filter_dim = filter_dim
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

        self.fact = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len], name='fact')
        self.fact_len = tf.placeholder(dtype=tf.int32, shape=[None], name='fact_len')
        self.law_kb = tf.placeholder(dtype=tf.int32, shape=[None, article_num, max_seq_len], name='law_kb')
        self.law_len = tf.placeholder(dtype=tf.int32, shape=[None, article_num], name='law_len')
        self.accu = tf.placeholder(dtype=tf.float32, shape=[None, accu_num], name='accu')
        self.article = tf.placeholder(dtype=tf.float32, shape=[None, article_num], name='article')

        with tf.variable_scope('fact_embedding'):
            fact_em = self.fact_embedding_layer()

        with tf.variable_scope('fact_encoder'):
            fact_enc = self.cnn_encoder(fact_em)

        with tf.variable_scope('article_extractor'):
            # art_score's shape = [batch_size, article_num]
            # top_k_score's shape = [batch_size, top_k]
            # top_k_indices' shape = [batch_size, top_k]
            self.art_score, top_k_score, top_k_indices = self.get_top_k_indices(fact_enc)

            # top_k_art's shape = [batch_size, top_k, max_seq_len]
            # top_k_art_len's shape = [batch_size, top_k]
            top_k_art, top_k_art_len = self.get_top_k_articles(top_k_indices)

        with tf.variable_scope('article_embedding'):
            top_k_art_em = self.article_embedding_layer(top_k_art)

        with tf.variable_scope('article_encoder', reuse=tf.AUTO_REUSE):
            art_em_splits = tf.split(top_k_art_em, self.top_k, axis=1)
            art_len_splits = tf.split(top_k_art_len, self.top_k, axis=1)

            art_enc_splits = []
            for art_em, art_len in zip(art_em_splits, art_len_splits):
                art_em = tf.reshape(art_em, [-1, self.max_seq_len, self.embedding_size])
                art_len = tf.reshape(art_len, [-1])

                art_enc = self.cnn_encoder(art_em)
                art_enc_splits.append(art_enc)

        with tf.variable_scope('attention_layer'):
            relevant_score = tf.nn.relu(top_k_score)
            score_splits = tf.split(relevant_score, self.top_k, axis=1)

            key = tf.layers.dense(
                fact_enc,
                self.att_size,
                tf.nn.tanh,
                use_bias=False,
                kernel_regularizer=self.regularizer
            )

            law_atts = []
            with tf.variable_scope('get_attention', reuse=tf.AUTO_REUSE):
                for art_enc, art_len, score in zip(art_enc_splits, art_len_splits, score_splits):
                    art_len = tf.reshape(art_len, [-1])
                    score = tf.reshape(score, [-1, 1, 1])

                    query = tf.layers.dense(
                        art_enc,
                        self.att_size,
                        tf.nn.tanh,
                        use_bias=False,
                        kernel_regularizer=self.regularizer
                    )

                    law_att = score * self.get_attention(query, key, art_len, self.fact_len)
                    law_atts.append(law_att)

            fact_enc_with_att = [tf.matmul(law_att, fact_enc) for law_att in law_atts]
            fact_enc_with_att = tf.reduce_max(tf.add_n(fact_enc_with_att), axis=-2)

        with tf.variable_scope('output_layer'):
            self.task_1_output, task_1_loss = self.output_layer(fact_enc_with_att, self.accu, self.accu_num)

        with tf.variable_scope('loss'):
            art_loss = tf.reduce_mean(tf.losses.hinge_loss(labels=self.article, logits=self.art_score))

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

    def rnn_encoder(self, inputs, seq_len):
        cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=seq_len,
            dtype=tf.float32
        )

        # output's shape = [batch_size, seq_len, 2 * hidden_size]
        output = tf.concat([output_fw, output_bw], axis=-1)

        return output

    def cnn_encoder(self, inputs):
        enc_output = []
        for kernel_size in self.kernel_size:
            with tf.variable_scope('conv_' + str(kernel_size)):
                # conv's shape = [batch_size, seq_len, filter_dim]
                conv = tf.layers.conv1d(
                    inputs,
                    filters=self.filter_dim,
                    kernel_size=kernel_size,
                    padding='same',
                    activation=tf.nn.relu,
                    kernel_regularizer=self.regularizer
                )

                enc_output.append(conv)

        # enc_output's shape = [batch_size, seq_len, len(kernel_size) * filter_dim]
        enc_output = tf.concat(enc_output, axis=-1)

        return enc_output

    def get_top_k_indices(self, inputs):
        inputs = tf.reduce_max(inputs, axis=-2)
        scores = tf.layers.dense(inputs, self.article_num, tf.nn.tanh, kernel_regularizer=self.regularizer)

        if self.is_training:
            top_k_score, top_k_indices = tf.math.top_k(self.article, k=self.top_k)
        else:
            top_k_score, top_k_indices = tf.math.top_k(scores, k=self.top_k)

        return scores, top_k_score, top_k_indices

    def get_top_k_articles(self, top_k_indices):
        art = tf.batch_gather(self.law_kb, indices=top_k_indices)
        art_len = tf.batch_gather(self.law_len, indices=top_k_indices)

        return art, art_len

    def article_embedding_layer(self, article):
        article_em = tf.nn.embedding_lookup(self.embedding_matrix, article)
        if self.is_training and self.keep_prob < 1.0:
            article_em = tf.nn.dropout(article_em, keep_prob=self.keep_prob)

        return article_em

    def get_attention(self, query, key, query_len, key_len):
        att = tf.nn.softmax(tf.matmul(query, key, transpose_b=True), axis=-1)

        mask_query = tf.sequence_mask(query_len, maxlen=self.max_seq_len, dtype=tf.float32)
        mask_key = tf.sequence_mask(key_len, maxlen=self.max_seq_len, dtype=tf.float32)
        mask = tf.matmul(tf.expand_dims(mask_query, axis=-1), tf.expand_dims(mask_key, axis=-2))

        # att's shape = [batch_size, seq_len, seq_len]
        att = mask * att

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
