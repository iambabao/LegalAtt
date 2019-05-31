import tensorflow as tf


class TopJudge:
    def __init__(self, config, embedding_matrix, is_training):
        self.accu_num = config.accu_num
        self.art_num = config.art_num
        self.impr_num = config.impr_num

        self.max_seq_len = config.sequence_len

        self.kernel_size = config.kernel_size
        self.filter_dim = config.filter_dim
        self.hidden_size = config.hidden_size
        self.fc_size = config.fc_size_s

        self.embedding_matrix = tf.get_variable(
            initializer=tf.constant_initializer(embedding_matrix),
            shape=embedding_matrix.shape,
            trainable=config.embedding_trainable,
            dtype=tf.float32,
            name='embedding_matrix'
        )
        self.embedding_size = embedding_matrix.shape[-1]

        self.lr = config.lr
        self.optimizer = config.optimizer
        self.dropout = config.dropout
        self.l2_rate = config.l2_rate
        self.use_batch_norm = config.use_batch_norm

        self.is_training = is_training

        self.w_init = tf.truncated_normal_initializer(stddev=0.1)
        self.b_init = tf.constant_initializer(0.1)

        if self.l2_rate > 0.0:
            self.regularizer = tf.keras.regularizers.l2(self.l2_rate)
        else:
            self.regularizer = None

        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.fact = tf.placeholder(dtype=tf.int32, shape=[None, self.max_seq_len], name='fact')
        self.fact_len = tf.placeholder(dtype=tf.int32, shape=[None], name='fact_len')
        self.art = tf.placeholder(dtype=tf.int32, shape=[None, self.art_num, self.max_seq_len], name='art')
        self.art_len = tf.placeholder(dtype=tf.int32, shape=[None, self.art_num], name='art_len')
        self.accu = tf.placeholder(dtype=tf.float32, shape=[None, self.accu_num], name='accu')
        self.relevant_art = tf.placeholder(dtype=tf.float32, shape=[None, self.art_num], name='relevant_art')
        self.impr = tf.placeholder(dtype=tf.float32, shape=[None, self.impr_num], name='impr')

        with tf.variable_scope('fact_embedding'):
            fact_em = self.embedding_layer(self.fact)

        with tf.variable_scope('fact_encoder'):
            fact_enc = self.cnn_encoder(fact_em)

        with tf.variable_scope('task_2'):
            _, task_2_state = self.lstm_encoder(fact_enc, None)

        with tf.variable_scope('task_1'):
            _, task_1_state = self.lstm_encoder(fact_enc, [task_2_state])

        with tf.variable_scope('task_3'):
            _, task_3_state = self.lstm_encoder(fact_enc, [task_1_state, task_2_state])

        with tf.variable_scope('output'):
            self.task_1_output, task_1_loss = self.output_layer(task_1_state[0], self.accu, layer='sigmoid')
            self.task_2_output, task_2_loss = self.output_layer(task_2_state[0], self.relevant_art, layer='sigmoid')
            self.task_3_output, task_3_loss = self.output_layer(task_3_state[0], self.impr, layer='softmax')

        with tf.variable_scope('loss'):
            self.loss = task_1_loss + task_2_loss + task_3_loss
            if self.regularizer is not None:
                l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self.loss += l2_loss

        if not is_training:
            return

        self.global_step, self.train_op = self.get_train_op()

    def embedding_layer(self, inputs):
        inputs_em = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        if self.is_training and self.dropout < 1.0:
            inputs_em = tf.nn.dropout(inputs_em, rate=self.dropout)

        return inputs_em

    def cnn_encoder(self, inputs):
        enc_output = []
        for kernel_size in self.kernel_size:
            conv = tf.keras.layers.Conv1D(
                self.filter_dim,
                kernel_size,
                padding='same',
                kernel_regularizer=self.regularizer,
                name='conv_' + str(kernel_size)
            )(inputs)
            if self.use_batch_norm:
                conv = tf.keras.layers.BatchNormalization(name='norm_' + str(kernel_size))(conv)
            conv = tf.nn.relu(conv)
            pool = tf.reduce_max(conv, axis=-2, keepdims=True)
            enc_output.append(pool)

        enc_output = tf.concat(enc_output, axis=-1)

        return enc_output

    def lstm_encoder(self, inputs, initial_state_list):
        lstm = tf.keras.layers.LSTM(self.hidden_size, return_state=True, name='lstm')

        if initial_state_list is not None:
            new_h = []
            new_c = []
            for h, c in initial_state_list:
                h = tf.keras.layers.Dense(self.hidden_size, kernel_regularizer=self.regularizer)(h)
                new_h.append(h)

                c = tf.keras.layers.Dense(self.hidden_size, kernel_regularizer=self.regularizer)(c)
                new_c.append(c)

            new_h = tf.math.add_n(new_h)
            new_c = tf.math.add_n(new_c)
            initial_state = [new_h, new_c]
        else:
            initial_state = lstm.get_initial_state(inputs)

        enc_output, enc_state_h, enc_state_c = lstm(inputs, initial_state)

        return enc_output, (enc_state_h, enc_state_c)

    def output_layer(self, inputs, labels, layer):
        fc_output = tf.keras.layers.Dense(self.fc_size, kernel_regularizer=self.regularizer)(inputs)
        if self.is_training and self.dropout < 1.0:
            fc_output = tf.nn.dropout(fc_output, rate=self.dropout)

        logits = tf.keras.layers.Dense(labels.shape[-1], kernel_regularizer=self.regularizer)(fc_output)
        if layer == 'softmax':
            output = tf.nn.softmax(logits)
            ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        elif layer == 'sigmoid':
            output = tf.nn.sigmoid(logits)
            ce_loss = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits),
                axis=-1
            ))
        else:
            assert False

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
            assert False

        train_op = optimizer.minimize(self.loss, global_step=global_step)

        return global_step, train_op
