import tensorflow as tf


class TopJudge(object):
    def __init__(self, accu_num, article_num, imprisonment_num,
                 max_seq_len, kernel_size, filter_dim, hidden_size, fc_size,
                 embedding_matrix, embedding_trainable,
                 lr, optimizer, keep_prob, l2_rate, use_batch_norm, is_training):
        self.accu_num = accu_num
        self.article_num = article_num
        self.imprisonment_num = imprisonment_num
        self.max_seq_len = max_seq_len

        self.kernel_size = kernel_size
        self.filter_dim = filter_dim
        self.hidden_size = hidden_size
        self.fc_size = fc_size

        self.embedding_matrix = tf.get_variable(
            initializer=tf.constant_initializer(embedding_matrix),
            shape=embedding_matrix.shape,
            trainable=embedding_trainable,
            dtype=tf.float32,
            name='embedding_matrix'
        )
        self.embedding_size = embedding_matrix.shape[-1]

        self.lr = lr
        self.optimizer = optimizer
        self.keep_prob = keep_prob
        self.l2_rate = l2_rate
        self.use_batch_norm = use_batch_norm

        self.is_training = is_training

        self.w_init = tf.truncated_normal_initializer(stddev=0.1)
        self.b_init = tf.constant_initializer(0.1)

        if l2_rate > 0.0:
            self.regularizer = tf.contrib.layers.l2_regularizer(l2_rate)
        else:
            self.regularizer = None

        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.fact = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len], name='fact')
        self.accu = tf.placeholder(dtype=tf.float32, shape=[None, accu_num], name='accu')
        self.article = tf.placeholder(dtype=tf.float32, shape=[None, article_num], name='article')
        self.imprisonment = tf.placeholder(dtype=tf.float32, shape=[None, imprisonment_num], name='imprisonment')

        with tf.variable_scope('fact_embedding'):
            fact_em = self.fact_embedding_layer()

        with tf.variable_scope('fact_encoder'):
            fact_enc = self.cnn_encoder(fact_em)

        with tf.variable_scope('task_2'):
            _, task_2_state = self.lstm_encoder(fact_enc, None)
            self.task_2_output, task_2_loss = self.output_layer(task_2_state.h, self.article, article_num)

        with tf.variable_scope('task_3'):
            _, task_3_state = self.lstm_encoder(fact_enc, [task_2_state])
            self.task_3_output, task_3_loss = self.output_layer(task_3_state.h, self.imprisonment, imprisonment_num)

        with tf.variable_scope('task_1'):
            _, task_1_state = self.lstm_encoder(fact_enc, [task_2_state, task_3_state])
            self.task_1_output, task_1_loss = self.output_layer(task_1_state.h, self.accu, accu_num)

        with tf.variable_scope('loss'):
            self.loss = task_1_loss + task_2_loss + task_3_loss
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

    def cnn_encoder(self, inputs):
        enc_output = []
        for kernel_size in self.kernel_size:
            with tf.variable_scope('conv_' + str(kernel_size)):
                conv = tf.layers.conv1d(
                    inputs,
                    filters=self.filter_dim,
                    kernel_size=kernel_size,
                    padding='same',
                    kernel_regularizer=self.regularizer
                )
                if self.use_batch_norm:
                    conv = tf.layers.batch_normalization(conv, training=self.is_training)
                conv = tf.nn.relu(conv)
                pool = tf.reduce_max(conv, axis=-2)

                enc_output.append(pool)

        enc_output = tf.concat(enc_output, axis=-1)

        return enc_output

    def lstm_encoder(self, inputs, initial_state_list):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        if initial_state_list is not None:
            new_c = []
            new_h = []
            for state in initial_state_list:
                c = tf.layers.dense(state.c, self.hidden_size, kernel_regularizer=self.regularizer)
                new_c.append(c)

                h = tf.layers.dense(state.h, self.hidden_size, kernel_regularizer=self.regularizer)
                new_h.append(h)

            new_c = tf.math.add_n(new_c)
            new_h = tf.math.add_n(new_h)
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
        else:
            initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        enc_output, enc_state = lstm_cell(inputs, initial_state)

        return enc_output, enc_state

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
        if self.use_batch_norm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group([train_op, update_ops])

        return global_step, train_op
