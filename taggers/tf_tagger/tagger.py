import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell, rnn, seq2seq


class Tagger(object):
    """LSTM tagger model."""
    def __init__(self, vocab, tagset, alphabet, word_embedding_size=128, char_embedding_size=16, batch_size=50, num_chars=20, num_steps=30, lr=0.1):

        self.word_embedding_size = word_embedding_size
        self.char_embedding_size = char_embedding_size

        self.lstm_size = lstm_size = word_embedding_size + 2 * char_embedding_size ###
        self.char_lstm_size = char_lstm_size = char_embedding_size # / 2 ### char_embedding_size 

        self.batch_size = batch_size

        self.vocab = vocab
        self.tagset = tagset
        self.alphabet = alphabet

        self.characters = tf.placeholder(tf.int32, [batch_size, num_steps, num_chars], name='characters')
        self.words = tf.placeholder(tf.int32, [batch_size, num_steps], name='words')
        self.sentence_lengths = tf.placeholder(tf.int64, [batch_size])
        self.tags = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.dropout_prob = tf.placeholder(tf.float32, [1])


        self.char_embeddings = tf.Variable(tf.random_uniform([len(alphabet), char_embedding_size], -1.0, 1.0))
        self.ce_lookup = tf.nn.embedding_lookup(self.char_embeddings, self.characters)

        self.embeddings = tf.Variable(tf.random_uniform([len(vocab), word_embedding_size], -1.0, 1.0))
        self.e_lookup = tf.nn.embedding_lookup(self.embeddings, self.words)

       
        self.char_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_chars, tf.reshape(self.ce_lookup, [batch_size * num_steps, num_chars, char_embedding_size], name="reshape-char_inputs"))]

        with tf.variable_scope('char_forward'):
            self.char_lstm = rnn_cell.BasicLSTMCell(char_embedding_size)
            self.char_outputs, self.char_states = rnn.rnn(
                cell=self.char_lstm,
                inputs=self.char_inputs,
                initial_state=self.char_lstm.zero_state(batch_size * num_steps, tf.float32))

        with tf.variable_scope('char_backward'):
            self.char_lstm_rev = rnn_cell.BasicLSTMCell(char_embedding_size)
            self.char_outputs_rev, self.char_states_rev = rnn.rnn(
                cell=self.char_lstm_rev,
                inputs=list(reversed(self.char_inputs)),
                initial_state=self.char_lstm_rev.zero_state(batch_size * num_steps, tf.float32))


        # self.char_outputs_bidi = [tf.concat(1, [o1, o2]) for o1, o2 in zip(self.char_outputs, reversed(self.char_outputs_rev))]
        
        # self.char_states je python list hodnot, ktery maj shape [batch_size * num_steps, char_embedding_size] a kterejch je num_chars.
        # my chceme vybrat posledni
        self.last_char_states = tf.reshape(self.char_outputs[num_chars-1], [batch_size, num_steps, char_embedding_size], name="reshape-charstates")
        self.last_char_states_rev = tf.reshape(self.char_outputs_rev[num_chars-1], [batch_size, num_steps, char_embedding_size], name="reshape-charstates_rev")

        self.char_output = tf.concat(2, [self.last_char_states, self.last_char_states_rev])

        ### input has 960,000 values which is not 48,000
        # ==> concat(1, char_outputs_bidi) ma 960000 hodnot
        # je to num_char-krat vic
        # nezajima nas cely pole char_outputs, zajima nas posledni hodnota. a mozna i posledni hodnota stavu, ne outputu

#        self.char_output = tf.reshape( self.char_bidi_state, [batch_size, num_steps, 2 * char_embedding_size], name="reshape-char_outputs_bidi")


        # inputs : seznam nakrajenej jako maslo
        self.inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, tf.concat(2, [self.char_output, self.e_lookup]))]

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
                    initial_state=self.lstm_rev.zero_state(batch_size, tf.float32))

        self.outputs_bidi = \
                [tf.concat(1, [o1, o2]) for o1, o2 in zip(self.outputs, reversed(self.outputs_rev))]

        #self.output = tf.reshape(tf.concat(1, self.outputs), [-1, self.lstm_size])
        self.output = tf.reshape(tf.concat(1, self.outputs_bidi), [-1, 2 * self.lstm_size], name="reshape-outputs_bidi")

#        self.output = tf.nn.dropout(self.output, self.dropout_prob[0])
        
        self.logits = tf.nn.xw_plus_b(
                self.output,
                tf.get_variable("softmax_w", [2 * self.lstm_size, len(tagset)]),
                tf.get_variable("softmax_b", [len(tagset)]))

        self.logits2 = tf.reshape(self.logits, [batch_size, num_steps, len(tagset)], name="reshape-logits")
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


    def learn(self, words, chars, tags, lengths):
        """Learn from the given minibatch."""

        fd = {self.words:words, self.characters:chars, self.tags:tags, self.sentence_lengths: lengths, self.dropout_prob: np.array([0.5])}
        _, cost = self.session.run([self.train, self.cost], feed_dict=fd)

        return cost


    def predict(self, words, chars, lengths):
        """Predict tags for the given minibatch."""

        logits = self.session.run(self.logits2, feed_dict={self.words: words, self.characters: chars, self.sentence_lengths: lengths, self.dropout_prob: np.array([1])})

        return np.argmax(logits, axis=2)
