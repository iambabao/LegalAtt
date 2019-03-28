import tensorflow as tf


class HAN(object):
    def __init__(self, accu_num, article_num, max_seq_len, max_doc_len,
                 hidden_size, att_size, fc_size,
                 embedding_matrix, embedding_trainable,
                 lr, optimizer, keep_prob, l2_rate, use_batch_norm, is_training):
        self.accu_num = accu_num
        self.article_num = article_num
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len

        self.hidden_size = hidden_size
        self.att_size = att_size
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

        self.fact = tf.placeholder(dtype=tf.int32, shape=[None, max_doc_len, max_seq_len], name='fact')
        self.fact_seq_len = tf.placeholder(dtype=tf.int32, shape=[None, max_doc_len], name='fact_seq_len')
        self.fact_doc_len = tf.placeholder(dtype=tf.int32, shape=[None], name='fact_doc_len')
        self.accu = tf.placeholder(dtype=tf.float32, shape=[None, accu_num], name='accu')
        self.article = tf.placeholder(dtype=tf.float32, shape=[None, article_num], name='article')

        with tf.variable_scope('fact_embedding'):
            fact_em = self.fact_embedding_layer()

        with tf.variable_scope('fact_encoder'):
            u_w = tf.get_variable(initializer=self.b_init, shape=[att_size], name='u_w')
            u_s = tf.get_variable(initializer=self.b_init, shape=[att_size], name='u_s')
            fact_enc = self.han_encoder(fact_em, self.fact_seq_len, self.fact_doc_len, u_w, u_s)

        with tf.variable_scope('output_layer'):
            self.task_1_output, task_1_loss = self.output_layer(fact_enc, self.accu, self.accu_num)
            self.task_2_output, task_2_loss = self.output_layer(fact_enc, self.article, self.article_num)

        with tf.variable_scope('loss'):
            self.loss = task_1_loss + task_2_loss
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

    def han_encoder(self, inputs, seq_len, doc_len, u_w, u_s):
        with tf.variable_scope('sequence_level'):
            inputs = tf.reshape(inputs, [-1, self.max_seq_len, self.embedding_size])
            seq_len = tf.reshape(seq_len, [-1])

            cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs,
                sequence_length=seq_len,
                dtype=tf.float32
            )

            seq_output = tf.concat([output_fw, output_bw], axis=-1)
            seq_output = tf.reshape(seq_output, [-1, self.max_doc_len, self.max_seq_len, 2 * self.hidden_size])
            if self.use_batch_norm:
                seq_output = tf.layers.batch_normalization(seq_output, training=self.is_training)

            u = tf.layers.dense(seq_output, self.att_size, tf.nn.tanh, kernel_regularizer=self.regularizer)
            u_att = tf.reshape(u_w, [-1, 1, 1, self.att_size])
            att_w = tf.nn.softmax(tf.reduce_sum(u * u_att, axis=-1), axis=-1)
            att_w = tf.reshape(att_w, [-1, self.max_doc_len, self.max_seq_len, 1])

            seq_output = tf.reduce_sum(att_w * seq_output, axis=-2)

        with tf.variable_scope('document_level'):
            doc_len = tf.reshape(doc_len, [-1])

            cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=seq_output,
                sequence_length=doc_len,
                dtype=tf.float32
            )

            doc_output = tf.concat([output_fw, output_bw], axis=-1)
            if self.use_batch_norm:
                doc_output = tf.layers.batch_normalization(doc_output, training=self.is_training)

            u = tf.layers.dense(doc_output, self.att_size, tf.nn.tanh, kernel_regularizer=self.regularizer)
            u_att = tf.reshape(u_s, [-1, 1, self.att_size])
            att_s = tf.nn.softmax(tf.reduce_sum(u * u_att, axis=-1), axis=-1)
            att_s = tf.reshape(att_s, [-1, self.max_doc_len, 1])

            doc_output = tf.reduce_sum(att_s * doc_output, axis=-2)

        return doc_output

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
