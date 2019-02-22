import tensorflow as tf
import math


class Transformer(object):
    def __init__(self, accu_num, article_num, imprisonment_num,
                 block_num, head_num, model_dim, fc_size,
                 embedding_matrix, embedding_trainable,
                 lr_base, lr_decay_rate, lr_decay_step, optimizer,
                 keep_prob, grad_clip, l2_rate, is_training):
        self.accu_num = accu_num
        self.article_num = article_num
        self.imprisonment_num = imprisonment_num

        self.block_num = block_num
        self.head_num = head_num
        self.model_dim = model_dim
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

        # fact_enc's shape = [batch_size, model_dim]
        with tf.variable_scope('fact_encoder'):
            fact_enc = self.fact_encoder(fact_em, self.fact_len)

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
        head_dim = self.model_dim / self.head_num
        cur_output = inputs
        for i in range(self.block_num):
            with tf.variable_scope('block_' + str(i)):
                pre_output = cur_output
                head_atts = [None] * self.head_num
                for j in range(self.head_num):
                    with tf.variable_scope('head_' + str(j)):
                        query = tf.layers.dense(pre_output, head_dim, kernel_regularizer=self.regularizer)
                        key = tf.layers.dense(pre_output, head_dim, kernel_regularizer=self.regularizer)
                        value = tf.layers.dense(pre_output, head_dim, kernel_regularizer=self.regularizer)

                        # mask's shape = [seq_len, seq_len]
                        mask = tf.sequence_mask(seq_len, dtype=tf.float32)
                        mask = tf.matmul(
                            tf.expand_dims(mask, axis=-1),
                            tf.expand_dims(mask, axis=-2)
                        )

                        weight = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / math.sqrt(head_dim), axis=-1)
                        weight = mask * weight

                        head_atts[j] = tf.matmul(weight, value)
                multi_head = tf.concat(head_atts, axis=-1)
                if self.is_training and self.keep_prob < 1.0:
                    multi_head = tf.nn.dropout(multi_head, keep_prob=self.keep_prob)
                multi_head = multi_head + pre_output

                # TODO: replace batch_norm with layer_norm
                multi_head = tf.layers.batch_normalization(multi_head, training=self.is_training)

                cur_output = tf.layers.dense(multi_head, self.fc_size, kernel_regularizer=self.regularizer)
                cur_output = tf.nn.relu(cur_output)
                cur_output = tf.layers.dense(cur_output, self.model_dim, kernel_regularizer=self.regularizer)
                if self.is_training and self.keep_prob < 1.0:
                    cur_output = tf.nn.dropout(cur_output, keep_prob=self.keep_prob)
                cur_output = cur_output + multi_head

                # TODO: replace batch_norm with layer_norm
                cur_output = tf.layers.batch_normalization(cur_output, training=self.is_training)

        final_output = tf.reduce_max(cur_output, axis=1)
        return final_output

    def output_layer(self, inputs, labels, task_id):
        if task_id == 'task_1':
            label_num = self.article_num
        elif task_id == 'task_2':
            label_num = self.imprisonment_num
        elif task_id == 'task_3':
            label_num = self.accu_num
        else:
            label_num = -1

        logits = tf.layers.dense(inputs, label_num, kernel_regularizer=self.regularizer)
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

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)
        return global_step, train_op
