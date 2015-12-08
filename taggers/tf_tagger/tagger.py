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
        self.tags = tf.placeholder(tf.int32, [batch_size, num_steps])

        self.embeddings = tf.Variable(tf.random_uniform([len(vocab), lstm_size], -1.0, 1.0))
        self.e_lookup = tf.nn.embedding_lookup(self.embeddings, self.words)

        self.lstm = rnn_cell.BasicLSTMCell(lstm_size)
        
        ## dyztak vrazit dropout

        # inputs : seznam nakrajenej jako maslo
        self.inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, self.e_lookup)]
        self.outputs, self.states = rnn.rnn(self.lstm, self.inputs, initial_state=self.lstm.zero_state(batch_size, tf.float32))

        self.output = tf.reshape(tf.concat(1, self.outputs), [-1, self.lstm_size])
        self.logits = tf.nn.xw_plus_b(self.output, tf.get_variable("softmax_w", [self.lstm_size, len(tagset)]), tf.get_variable("softmax_b", [len(tagset)]))
        self.logits2 = tf.reshape(self.logits, [batch_size, num_steps, len(tagset)])
#        self.tags_hat = tf.argmax(self.logits, 2)

        ## ty jednicky sou vahy
        self.loss = seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.tags, [-1])], [tf.ones([batch_size * num_steps])], len(tagset))
        self.cost = tf.reduce_sum(self.loss) / batch_size

        self.lr = lr
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train = self.optimizer.minimize(self.cost)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())                        



    def learn(self, mb_x, mb_y):
        """Learn from the given minibatch."""

        # self.Session

        fd = {self.words:mb_x, self.tags:mb_y}
        _, cost = self.session.run([self.train, self.cost], feed_dict=fd)       

        return cost





    def predict(self, mb_x):
        """Predict tags for the given minibatch."""
        
        logits = self.session.run(self.logits2, feed_dict={self.words: mb_x})

#        import sys
#        print >> sys.stderr, logits.shape


        return np.argmax(logits, axis=2)
