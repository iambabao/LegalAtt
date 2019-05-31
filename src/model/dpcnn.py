import tensorflow as tf
import math


class DPCNN:
    def __init__(self, config, embedding_matrix, is_training):
        self.accu_num = config.accu_num
        self.art_num = config.art_num
        self.impr_num = config.impr_num

        self.max_seq_len = config.sequence_len

        self.kernel_size = config.kernel_size[0]
        self.filter_dim = config.filter_dim
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
            fact_enc = self.dpcnn_encoder(fact_em, self.max_seq_len)

        with tf.variable_scope('output'):
            self.task_1_output, task_1_loss = self.output_layer(fact_enc, self.accu, layer='sigmoid')
            self.task_2_output, task_2_loss = self.output_layer(fact_enc, self.relevant_art, layer='sigmoid')

        with tf.variable_scope('loss'):
            self.loss = task_1_loss + task_2_loss
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

    def dpcnn_encoder(self, inputs, max_seq_len):
        output_len = max_seq_len
        cur_output = tf.keras.layers.Dense(self.filter_dim, kernel_regularizer=self.regularizer)(inputs)
        while output_len > 1:
            pre_output = cur_output

            cur_block = tf.keras.layers.Conv1D(
                self.filter_dim,
                self.kernel_size,
                padding='same',
                kernel_regularizer=self.regularizer,
                name='conv_a_' + str(output_len)
            )(pre_output)
            if self.use_batch_norm:
                cur_block = tf.keras.layers.BatchNormalization(name='norm_a_' + str(output_len))(cur_block)
            cur_block = tf.nn.relu(cur_block)

            cur_block = tf.keras.layers.Conv1D(
                self.filter_dim,
                self.kernel_size,
                padding='same',
                kernel_regularizer=self.regularizer,
                name='conv_b_' + str(output_len)
            )(cur_block)
            if self.use_batch_norm:
                cur_block = tf.keras.layers.BatchNormalization(name='norm_b_' + str(output_len))(cur_block)
            cur_block = tf.nn.relu(cur_block)

            cur_output = tf.add(cur_block, pre_output)
            cur_output = tf.keras.layers.MaxPooling1D(self.kernel_size, strides=2, padding='same')(cur_output)

            output_len = math.ceil(output_len / 2)

        cur_output = tf.reshape(cur_output, [-1, self.filter_dim])
        return cur_output

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
