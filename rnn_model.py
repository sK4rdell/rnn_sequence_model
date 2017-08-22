import tensorflow as tf


class RNNSeqModel():
    def __init__(self, num_hidden, num_layers, num_chars, lr=1e-4):
        self.placeholders = {}
        with tf.name_scope('placeholders'):
            self.placeholders['y'] = tf.placeholder(dtype=tf.uint8, shape=[None, None], name='y')
            self.placeholders['x'] = tf.placeholder(tf.uint8, [None, None], name='x')
            self.placeholders['lstm_state'] = tf.placeholder(tf.float32,
                                                             [num_layers, 2, None, num_hidden], name='lstm_state')
            self.placeholders['keep_prob'] = tf.placeholder(tf.float32, name='keep_prob')

        self.output = self._build_model(num_hidden, num_layers, num_chars)

        with tf.name_scope('Loss'):
            # make 1-hot vector for labels
            labels = tf.one_hot(self.placeholders['y'], num_chars, 1., 0., name='1_hot_label')
            labels = tf.reshape(labels, shape=[-1, num_chars])

            x_ent = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=self.output['logits'])
            self.loss = tf.reduce_mean(x_ent)

        with tf.name_scope('Optimizer'):
            self._trainer = tf.train.AdamOptimizer(lr)
            self.update = self._trainer.minimize(self.loss)


    def _build_model(self, num_hidden, num_layers, num_chars):

        # cumbersome way to handle LSTM-state tuples
        l = tf.unstack(self.placeholders['lstm_state'], axis=0)
        stacked_lstm_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
                                    for idx in range(num_layers)])

        x_one_hot = tf.one_hot(self.placeholders['x'], num_chars, 1., 0., name='1_hot_x')

        # create stacked lstm-cells
        cells = [tf.contrib.rnn.LSTMCell(num_hidden) for _ in range(num_layers)]
        cells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.placeholders['keep_prob']) for cell in cells]
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)
        stacked_lstm = tf.contrib.rnn.DropoutWrapper(stacked_lstm, output_keep_prob=self.placeholders['keep_prob'])

        lstm_out, state_out = tf.nn.dynamic_rnn(stacked_lstm,
                                                x_one_hot,
                                                dtype=tf.float32,
                                                initial_state=stacked_lstm_state)

        flat_out = tf.reshape(lstm_out, shape=[-1, num_hidden], name='flat_lstm_out')

        logits = tf.layers.dense(inputs=flat_out,
                                 activation=tf.identity,
                                 units=num_chars,
                                 use_bias=False,
                                 name='fc_logits')

        cat_dist = tf.nn.softmax(logits, name='cat_dist')

        output_dict = {'lstm_state_out': state_out,
                       'logits': logits,
                       'cat_dist': cat_dist}

        return output_dict
