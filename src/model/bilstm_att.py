import tensorflow as tf


class BiLSTMAtt(object):
    def __init__(self, accu_num, article_num, imprisonment_num,
                 hidden_size, att_size, fc_size,
                 embedding_matrix, embedding_trainable,
                 lr_base, lr_decay_rate, lr_decay_step, optimizer,
                 keep_prob, grad_clip, l2_rate, is_training):
        self.accu_num = accu_num
        self.article_num = article_num
        self.imprisonment_num = imprisonment_num

        self.hidden_size = hidden_size
        self.att_size = att_size
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
        self.fact_len = tf.placeholder(dtype=tf.int32, shape=[None], name='fact_len')
        self.accu = tf.placeholder(dtype=tf.float32, shape=[None, accu_num], name='accu')
        self.article = tf.placeholder(dtype=tf.float32, shape=[None, article_num], name='article')
        self.imprisonment = tf.placeholder(dtype=tf.int32, shape=[None, imprisonment_num], name='imprisonment')

        # fact_em's shape = [batch_size, seq_len, embedding_size]
        with tf.variable_scope('fact_embedding'):
            fact_em = self.fact_embedding_layer()

        # fact_enc's shape = [batch_size, 2 * hidden_size]
        with tf.variable_scope('fact_encoder'):
            u_att = tf.get_variable(initializer=self.b_init, shape=[att_size], name='u_att')
            fact_enc = self.fact_encoder(fact_em, self.fact_len, u_att)

        with tf.variable_scope('task_1'):
            self.task_1_output, self.task_1_loss = self.output_layer(fact_enc, self.article, 'task_1')

        with tf.variable_scope('task_2'):
            self.task_2_output, self.task_2_loss = self.output_layer(fact_enc, self.imprisonment, 'task_2')

        with tf.variable_scope('task_3'):
            self.task_3_output, self.task_3_loss = self.output_layer(fact_enc, self.accu, 'task_3')

        self.loss = self.task_3_loss
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

    def fact_encoder(self, inputs, seq_len, u_att):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=seq_len,
            dtype=tf.float32
        )
        # output's shape = [batch_size, seq_len, 2 * hidden_size]
        output = tf.concat([output_fw, output_bw], axis=-1)

        # att's shape = [batch_size, seq_len, 1]
        u = tf.math.tanh(tf.layers.dense(output, self.att_size, kernel_regularizer=self.regularizer))
        u_att = tf.reshape(u_att, [-1, 1, self.att_size])
        mask = tf.sequence_mask(seq_len, dtype=tf.float32)
        att = mask * tf.math.softmax(tf.reduce_sum(u * u_att, axis=-1), axis=-1)
        att = tf.expand_dims(att, axis=-1)

        # output's shape = [batch_size, 2 * hidden_size]
        output = tf.reduce_sum(att * output, axis=1)
        return output

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
