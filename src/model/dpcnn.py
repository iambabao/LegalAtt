import tensorflow as tf
import math


class DPCNN(object):
    def __init__(self, accu_num, article_num, imprisonment_num,
                 seq_len, filter_size, filter_dim, fc_size,
                 embedding_matrix, embedding_trainable,
                 lr_base, lr_decay_rate, lr_decay_step, optimizer,
                 keep_prob, grad_clip, l2_rate, is_training):
        self.accu_num = accu_num
        self.article_num = article_num
        self.imprisonment_num = imprisonment_num

        self.filter_size = filter_size
        self.filter_dim = filter_dim
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

        self.fact = tf.placeholder(dtype=tf.int32, shape=[None, seq_len], name='fact')
        self.accu = tf.placeholder(dtype=tf.float32, shape=[None, accu_num], name='accu')
        self.article = tf.placeholder(dtype=tf.float32, shape=[None, article_num], name='article')
        self.imprisonment = tf.placeholder(dtype=tf.int32, shape=[None, imprisonment_num], name='imprisonment')

        # fact_em's shape = [batch_size, seq_len, embedding_size]
        with tf.variable_scope('fact_embedding'):
            fact_em = self.fact_embedding_layer()

        # fact_enc's shape = [batch_size, filter_dim]
        with tf.variable_scope('fact_encoder'):
            fact_enc = self.fact_encoder(fact_em, seq_len)

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

    def fact_encoder(self, inputs, seq_len):
        output_len = seq_len
        cur_output = tf.layers.dense(inputs, self.filter_dim, kernel_regularizer=self.regularizer)
        while output_len > 1:
            pre_output = cur_output
            cur_block = tf.nn.relu(pre_output)
            cur_block = tf.layers.conv1d(cur_block, self.filter_dim, self.filter_size,
                                         padding='same', kernel_regularizer=self.regularizer)
            cur_block = tf.nn.relu(cur_block)
            cur_block = tf.layers.conv1d(cur_block, self.filter_dim, self.filter_size,
                                         padding='same', kernel_regularizer=self.regularizer)
            cur_output = tf.add(cur_block, pre_output)
            cur_output = tf.layers.max_pooling1d(cur_output, pool_size=3, strides=2, padding='same')
            output_len = math.ceil(output_len / 2)

        cur_output = tf.reshape(cur_output, [-1, self.filter_dim])
        return cur_output

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
