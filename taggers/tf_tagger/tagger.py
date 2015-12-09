import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell, rnn, seq2seq


class Tagger(object):
    """LSTM tagger model."""
    def __init__(self, vocab, tagset, lstm_size=128, batch_size=50, num_steps=30, lr=0.1):
        self.lstm_size = lstm_size
        self.batch_size = batch_size

        self.vocab = vocab
        self.tagset = tagset

        self.words = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.sentence_lengths = tf.placeholder(tf.int64, [batch_size])
        self.tags = tf.placeholder(tf.int32, [batch_size, num_steps])

        self.embeddings = tf.Variable(tf.random_uniform([len(vocab), lstm_size], -1.0, 1.0))
        self.e_lookup = tf.nn.embedding_lookup(self.embeddings, self.words)


        ## dyztak vrazit dropout

        # inputs : seznam nakrajenej jako maslo
        self.inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, self.e_lookup)]

        with tf.variable_scope('forward'):
            self.lstm = rnn_cell.BasicLSTMCell(lstm_size)
            self.outputs, self.states = rnn.rnn(
                    cell=self.lstm,
                    inputs=self.inputs,
                    initial_state=self.lstm.zero_state(batch_size, tf.float32))#,
                    #sequence_length=self.sentence_lengths)

        with tf.variable_scope('backward'):
            self.lstm_rev = rnn_cell.BasicLSTMCell(lstm_size)
            self.outputs_rev, self.states_rev = rnn.rnn(
                    cell=self.lstm_rev,
                    inputs=list(reversed(self.inputs)),
                    initial_state=self.lstm.zero_state(batch_size, tf.float32))

        self.outputs_bidi = \
                [tf.concat(1, [o1, o2]) for o1, o2 in zip(self.outputs, reversed(self.outputs_rev))]

        #self.output = tf.reshape(tf.concat(1, self.outputs), [-1, self.lstm_size])
        self.output = tf.reshape(tf.concat(1, self.outputs_bidi), [-1, 2 * self.lstm_size])
        self.logits = tf.nn.xw_plus_b(
                self.output,
                tf.get_variable("softmax_w", [2 * self.lstm_size, len(tagset)]),
                tf.get_variable("softmax_b", [len(tagset)]))

        self.logits2 = tf.reshape(self.logits, [batch_size, num_steps, len(tagset)])
#        self.tags_hat = tf.argmax(self.logits, 2)

        # output maks: compute loss only if it insn't a padded word (i.e. zero index)
        #output_mask = tf.reshape(tf.to_float(tf.not_equal(self.words, 0)), [-1])
        output_mask = tf.ones([batch_size * num_steps])
        self.loss = seq2seq.sequence_loss_by_example(
            logits=[self.logits],
            targets=[tf.reshape(self.tags, [-1])],
            weights=[output_mask],
            num_decoder_symbols=len(tagset))
        self.cost = tf.reduce_sum(self.loss) / batch_size

        self.lr = lr
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train = self.optimizer.minimize(self.cost)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())


    def learn(self, mb_x, mb_y, lengths):
        """Learn from the given minibatch."""

        fd = {self.words:mb_x, self.tags:mb_y, self.sentence_lengths: lengths}
        _, cost = self.session.run([self.train, self.cost], feed_dict=fd)

        return cost


    def predict(self, mb_x, lengths):
        """Predict tags for the given minibatch."""

        logits = self.session.run(self.logits2, feed_dict={self.words: mb_x, self.sentence_lengths: lengths})

        return np.argmax(logits, axis=2)
