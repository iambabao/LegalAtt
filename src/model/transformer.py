import tensorflow as tf
import math


class Transformer(object):
    def __init__(self, accu_num, max_seq_len,
                 block_num, head_num, model_dim, fc_size,
                 embedding_matrix, embedding_trainable,
                 lr, optimizer, keep_prob, l2_rate, is_training):
        self.accu_num = accu_num

        self.max_seq_len = max_seq_len
        self.block_num = block_num
        self.head_num = head_num
        self.model_dim = model_dim
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
        self.accu = tf.placeholder(dtype=tf.float32, shape=[None, accu_num], name='accu')

        # fact_em's shape = [batch_size, seq_len, embedding_size]
        with tf.variable_scope('fact_embedding'):
            fact_em = self.fact_embedding_layer()

        # fact_enc's shape = [batch_size, model_dim]
        with tf.variable_scope('fact_encoder'):
            fact_enc = self.fact_encoder(fact_em, self.fact_len)

        with tf.variable_scope('output_layer'):
            self.task_1_output, task_1_loss = self.output_layer(fact_enc, self.accu, self.accu_num)

        with tf.variable_scope('loss'):
            self.loss = task_1_loss
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
        head_dim = self.model_dim / self.head_num
        cur_output = inputs
        for i in range(self.block_num):
            with tf.variable_scope('block_' + str(i)):
                pre_output = cur_output
                # head_atts'shape = [batch_size, max_seq_len, head_dim] * head_num
                head_atts = []
                for j in range(self.head_num):
                    with tf.variable_scope('head_' + str(j)):
                        # shape = [batch_size, max_seq_len, head_dim]
                        query = tf.layers.dense(pre_output, head_dim, kernel_regularizer=self.regularizer)
                        key = tf.layers.dense(pre_output, head_dim, kernel_regularizer=self.regularizer)
                        value = tf.layers.dense(pre_output, head_dim, kernel_regularizer=self.regularizer)

                        # mask's shape = [batch_size, max_seq_len, max_seq_len]
                        mask = tf.sequence_mask(seq_len, maxlen=self.max_seq_len, dtype=tf.float32)
                        mask = tf.matmul(
                            tf.expand_dims(mask, axis=-1),
                            tf.expand_dims(mask, axis=-2)
                        )

                        # weight's shape = [batch_size, max_seq_len, max_seq_len]
                        weight = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / math.sqrt(head_dim), axis=-1)
                        weight = mask * weight

                        head_atts.append(tf.matmul(weight, value))

                # multi_head'shape = [batch_size, max_seq_len, model_dim]
                multi_head = tf.concat(head_atts, axis=-1)
                if self.is_training and self.keep_prob < 1.0:
                    multi_head = tf.nn.dropout(multi_head, keep_prob=self.keep_prob)
                multi_head = multi_head + pre_output

                # TODO: replace batch_norm with layer_norm
                multi_head = tf.layers.batch_normalization(multi_head, training=self.is_training)

                # cur_output's shape = [batch_size, max_seq_len, model_dim]
                cur_output = tf.layers.dense(multi_head, self.fc_size, tf.nn.relu, kernel_regularizer=self.regularizer)
                cur_output = tf.layers.dense(cur_output, self.model_dim, kernel_regularizer=self.regularizer)
                if self.is_training and self.keep_prob < 1.0:
                    cur_output = tf.nn.dropout(cur_output, keep_prob=self.keep_prob)
                cur_output = cur_output + multi_head

                # TODO: replace batch_norm with layer_norm
                cur_output = tf.layers.batch_normalization(cur_output, training=self.is_training)

        final_output = tf.reduce_max(cur_output, axis=-2)
        return final_output

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

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss, global_step=global_step)
        train_op = tf.group([train_op, update_ops])

        # train_op = optimizer.minimize(self.loss, global_step=global_step)
        return global_step, train_op
