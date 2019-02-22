import tensorflow as tf


class TopJudge(object):
    def __init__(self, accu_num, article_num, imprisonment_num,
                 filter_size, filter_dim, hidden_size, fc_size,
                 embedding_matrix, embedding_trainable,
                 lr_base, lr_decay_rate, lr_decay_step, optimizer,
                 keep_prob, grad_clip, l2_rate, is_training):
        self.accu_num = accu_num
        self.article_num = article_num
        self.imprisonment_num = imprisonment_num

        self.filter_size = filter_size
        self.filter_dim = filter_dim
        self.hidden_size = hidden_size
        self.fc_size = fc_size

        self.embedding_matrix = tf.get_variable(
            initializer=tf.constant_initializer(embedding_matrix),
            shape=embedding_matrix.shape,
            trainable=embedding_trainable,
            name='embedding_matrix')
        self.embedding_size = embedding_matrix.shape[-1]

        self.lr_base = lr_base
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.optimizer = optimizer

        self.keep_prob = keep_prob
        self.grad_clip = grad_clip
        self.l2_rate = l2_rate

        self.is_training = is_training

        self.w_init = tf.truncated_normal_initializer(stddev=0.1)
        self.b_init = tf.constant_initializer(0.1)

        if l2_rate > 0.0:
            self.regularizer = tf.contrib.layers.l2_regularizer(l2_rate)
        else:
            self.regularizer = None

        self.fact = tf.placeholder(dtype=tf.int32, shape=[None, None], name='fact')
        self.accu = tf.placeholder(dtype=tf.float32, shape=[None, accu_num], name='accu')
        self.article = tf.placeholder(dtype=tf.float32, shape=[None, article_num], name='article')
        self.imprisonment = tf.placeholder(dtype=tf.int32, shape=[None, imprisonment_num], name='imprisonment')

        self.batch_size = tf.shape(self.fact)[0]

        # fact_em's shape = [batch_size, seq_len, embedding_size]
        with tf.variable_scope('fact_embedding'):
            fact_em = self.fact_embedding_layer()

        # fact_enc's shape = [batch_size, filter_dim, len(filter_size)]
        with tf.variable_scope('fact_encoder'):
            fact_enc = self.fact_encoder(fact_em)

        # task 1: predict relevant articles
        # task 2: predict term of imprisonment
        # task 3: predict accusation
        # task 3 is based on task 1 and task 2
        # enc_outputs' shape = [batch_size, filter_dim, hidden_size]
        # enc_state's shape = [batch_size, hidden_size] * 2
        with tf.variable_scope('task_1'):
            task_1_outputs, task_1_state = self.lstm_encoder(fact_enc, None)
            self.task_1_output, self.task_1_loss = self.output_layer(task_1_state.h, self.article, 'task_1')

        with tf.variable_scope('task_2'):
            task_2_outputs, task_2_state = self.lstm_encoder(fact_enc, None)
            self.task_2_output, self.task_2_loss = self.output_layer(task_2_state.h, self.imprisonment, 'task_2')

        with tf.variable_scope('task_3'):
            task_3_outputs, task_3_state = self.lstm_encoder(fact_enc, [task_1_state, task_2_state])
            self.task_3_output, self.task_3_loss = self.output_layer(task_3_state.h, self.accu, 'task_3')

        self.loss = self.task_1_loss + self.task_2_loss + self.task_3_loss
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

    def fact_encoder(self, inputs):
        enc_output = None
        for filter_size in self.filter_size:
            with tf.variable_scope('conv_' + str(filter_size)):
                conv_w = tf.get_variable(
                    initializer=self.w_init, shape=[filter_size, self.embedding_size, self.filter_dim], name='conv_w'
                )
                conv_b = tf.get_variable(
                    initializer=self.b_init, shape=self.filter_dim, name='conv_b'
                )

                # conv's shape = [batch_size, seq_len, filter_dim]
                conv = tf.nn.relu(tf.nn.bias_add(
                    tf.nn.conv1d(inputs, conv_w, stride=1, padding='SAME'), conv_b
                ))
                # pool's shape = [batch_size, filter_dim, 1]
                pool = tf.expand_dims(tf.reduce_max(conv, 1), -1)
                if enc_output is None:
                    enc_output = pool
                else:
                    enc_output = tf.concat([enc_output, pool], axis=-1)

                if self.regularizer is not None:
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.regularizer(conv_w))

        return enc_output

    def lstm_encoder(self, inputs, initial_state_list):
        if initial_state_list is not None:
            c_sum = tf.zeros(shape=[self.batch_size, self.hidden_size], dtype=tf.float32)
            h_sum = tf.zeros(shape=[self.batch_size, self.hidden_size], dtype=tf.float32)
            for i, state in enumerate(initial_state_list):
                c = tf.layers.dense(state.c, self.hidden_size, kernel_regularizer=self.regularizer)
                c_sum += c

                h = tf.layers.dense(state.h, self.hidden_size, kernel_regularizer=self.regularizer)
                h_sum += h

            initial_state = tf.nn.rnn_cell.LSTMStateTuple(c_sum, h_sum)
        else:
            initial_state = None

        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        enc_outputs, enc_state = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=inputs,
            initial_state=initial_state,
            dtype=tf.float32
        )

        return enc_outputs, enc_state

    def output_layer(self, inputs, labels, task_id):
        if task_id == 'task_1':
            label_num = self.article_num
        elif task_id == 'task_2':
            label_num = self.imprisonment_num
        elif task_id == 'task_3':
            label_num = self.accu_num
        else:
            label_num = -1

        fc_output = tf.layers.dense(inputs, self.fc_size, kernel_regularizer=self.regularizer)
        if self.is_training and self.keep_prob < 1.0:
            fc_output = tf.nn.dropout(fc_output, keep_prob=self.keep_prob)
        logits = tf.layers.dense(fc_output, label_num, kernel_regularizer=self.regularizer)
        output = tf.nn.softmax(logits)

        ce_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        )

        return output, ce_loss

    def get_train_op(self):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr = tf.train.exponential_decay(
            learning_rate=self.lr_base,
            decay_rate=self.lr_decay_rate,
            decay_steps=self.lr_decay_step,
            global_step=global_step
        )
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        if self.grad_clip > 0.0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)

        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        elif self.optimizer == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
        elif self.optimizer == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
        elif self.optimizer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        train_op = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)
        return global_step, train_op
