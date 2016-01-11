import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell, rnn, seq2seq


class Tagger(object):
    """LSTM tagger model."""
    def __init__(self, vocab, tagset, alphabet, word_embedding_size,
                 char_embedding_size, num_chars, num_steps, optimizer_desc,
                 generate_lemmas, seed=None, write_summaries=False):
        """
        Builds the tagger computation graph and initializes it in a TensorFlow
        session.

        Arguments:

            vocab: Vocabulary of word forms.

            tagset: Vocabulary of possible tags.

            alphabet: Vocabulary of possible characters.

            word_embedding_size (int): Size of the form-based word embedding.

            char_embedding_size (int): Size of character embeddings, i.e. a
                half of the size of the character-based words embeddings.

            num_chars: Maximum length of a word.

            num_steps: Maximum lenght of a sentence.

            optimizer_desc: Description of the optimizer.

            generate_lemmas: Generate lemmas during tagging.

            seed: TensorFlow seed

            write_summaries: Write summaries using TensorFlow interface.
        """

        self.num_steps = num_steps
        self.num_chars = num_chars

        self.word_embedding_size = word_embedding_size
        self.char_embedding_size = char_embedding_size

        self.lstm_size = lstm_size = word_embedding_size + 2 * char_embedding_size ###
        self.char_lstm_size = char_lstm_size = char_embedding_size # / 2 ### char_embedding_size

        self.vocab = vocab
        self.tagset = tagset
        self.alphabet = alphabet

        self.forward_initial_state = tf.placeholder(tf.float32, [None, rnn_cell.BasicLSTMCell(lstm_size).state_size])
        self.backward_initial_state = tf.placeholder(tf.float32, [None, rnn_cell.BasicLSTMCell(lstm_size).state_size])
        self.sentence_lengths = tf.placeholder(tf.int64, [None])
        self.tags = tf.placeholder(tf.int32, [None, num_steps])
        self.dropout_prob = tf.placeholder(tf.float32, [1])

        self.input_list = []

        # Character-level embeddings
        if char_embedding_size:
            self.characters = tf.placeholder(tf.int32, [None, num_steps, num_chars], name='characters')

            self.char_embeddings = \
                tf.Variable(tf.random_uniform([len(alphabet), char_embedding_size], -1.0, 1.0))
            self.ce_lookup = tf.nn.embedding_lookup(self.char_embeddings, self.characters)

            self.char_inputs = \
                [tf.squeeze(input_, [1]) for input_ in
                        tf.split(1, num_chars, tf.reshape(self.ce_lookup,
                            [-1, num_chars, char_embedding_size], name="reshape-char_inputs"))]
                             # ^^ -1 = batch_size * num_steps

            with tf.variable_scope('char_forward'):
                self.char_lstm = rnn_cell.BasicLSTMCell(char_embedding_size)
                self.char_outputs, self.char_states = rnn.rnn(
                    cell=self.char_lstm,
                    inputs=self.char_inputs, dtype=tf.float32)
                    #initial_state=self.char_lstm.zero_state(-1, tf.float32))

            with tf.variable_scope('char_backward'):
                self.char_lstm_rev = rnn_cell.BasicLSTMCell(char_embedding_size)
                self.char_outputs_rev, self.char_states_rev = rnn.rnn(
                    cell=self.char_lstm_rev,
                    inputs=list(reversed(self.char_inputs)), dtype=tf.float32)

            self.last_char_lstm_state = tf.split(1, 2, self.char_states[num_chars - 1])[1]
            self.last_char_lstm_state_rev = tf.split(1, 2, self.char_states_rev[num_chars - 1])[1]

            self.last_char_states = \
                tf.reshape(self.last_char_lstm_state, [-1, num_steps, char_embedding_size],
                           name="reshape-charstates")
            self.last_char_states_rev = tf.reshape(self.last_char_lstm_state_rev, [-1, num_steps, char_embedding_size], name="reshape-charstates_rev")

            self.char_output = tf.concat(2, [self.last_char_states, self.last_char_states_rev])

            self.input_list.append(self.char_output)

        # Word-level embeddings
        if word_embedding_size:
            self.words = tf.placeholder(tf.int32, [None, num_steps], name='words')
            self.embeddings = tf.Variable(tf.random_uniform([len(vocab), word_embedding_size], -1.0, 1.0))
            self.e_lookup = tf.nn.embedding_lookup(self.embeddings, self.words)

            self.input_list.append(self.e_lookup)

        # All inputs correctly sliced
        self.inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, tf.concat(2, self.input_list))]

        with tf.variable_scope('forward'):
            self.lstm = rnn_cell.BasicLSTMCell(lstm_size)
            self.outputs, self.states = rnn.rnn(
                    cell=self.lstm,
                    inputs=self.inputs, dtype=tf.float32,
                    initial_state=self.forward_initial_state)#,
                    #sequence_length=self.sentence_lengths)

        with tf.variable_scope('backward'):
            self.lstm_rev = rnn_cell.BasicLSTMCell(lstm_size)
            self.outputs_rev, self.states_rev = rnn.rnn(
                    cell=self.lstm_rev,
                    inputs=list(reversed(self.inputs)), dtype=tf.float32,
                    initial_state=self.backward_initial_state)

        self.outputs_bidi = \
                [tf.concat(1, [o1, o2]) for o1, o2 in zip(self.outputs, reversed(self.outputs_rev))]

        self.output = \
            tf.reshape(tf.concat(1, self.outputs_bidi), [-1, 2 * self.lstm_size],
                       name="reshape-outputs_bidi")

        # We are computing only the logits, not the actual softmax -- while
        # computing the loss, it is done by the sequence_loss_by_example and
        # during the runtime classification, the argmax over logits is enough.

        self.logits_flatten = tf.nn.xw_plus_b(
                self.output,
                tf.get_variable("softmax_w", [2 * self.lstm_size, len(tagset)]),
                tf.get_variable("softmax_b", [len(tagset)]))

        self.logits = tf.reshape(self.logits_flatten, [-1, num_steps, len(tagset)], name="reshape-logits")

        # output maks: compute loss only if it insn't a padded word (i.e. zero index)
        output_mask = tf.reshape(tf.to_float(tf.not_equal(self.tags, 0)), [-1])

        self.tagging_loss = seq2seq.sequence_loss_by_example(
            logits=[self.logits_flatten],
            targets=[tf.reshape(self.tags, [-1])],
            weights=[output_mask],
            num_decoder_symbols=len(tagset))

        self.cost = tf.reduce_mean(self.tagging_loss)

        if generate_lemmas:
            with tf.variable_scope('decoder'):
                self.lemma_characters = tf.placeholder(tf.int32, [None, num_steps, num_chars], name='lemma_characters')

                self.lemma_state_size = 2 * self.lstm_size

                self.lemma_W = tf.Variable(tf.random_uniform([self.lemma_state_size, len(self.alphabet)], 0.5),
                        name="state_to_char_W")
                self.lemma_B = \
                    tf.Variable(tf.fill([len(self.alphabet)], - math.log(len(self.alphabet))),
                            name="state_to_char_b")
                self.lemma_char_embeddings = \
                    tf.Variable(tf.random_uniform([len(self.alphabet), self.lemma_state_size], -0.5, 0.5),
                            name="char_embeddings")

                self.lemma_char_inputs = \
                    [tf.squeeze(input_, [1]) for input_ in
                            tf.split(1, num_chars, tf.reshape(self.lemma_characters,
                                [-1, num_chars, len(self.alphabet)], name="reshape-lemma_char_inputs"))]

                def loop(prev_state, _):
                    # it takes the previous hidden state, finds the character and formats it
                    # as input for the next time step ... used in the decoder in the "real decoding scenario"
                    out_activation = tf.matmul(prev_state, self.lemma_W) + self.lemma_B
                    prev_char_index = tf.argmax(out_activation, 1)
                    return tf.nn.embedding_lookup(self.lemma_char_embeddings, prev_char_index)

                embedded_lemma_characters = []
                for lemma_chars in self.lemma_char_inputs[:-1]:
                    embedded_lemma_characters.append(tf.nn.embedding_lookup(self.lemma_char_embeddings, lemma_chars))

                decoder_cell = rnn_cell.BasicLSTMCell(self.lemma_state_size)
                self.lemma_outputs_train, _ = seq2seq.rnn_decoder(embedded_lemma_characters, self.output,
                                                                  decoder_cell)

                tf.get_variable_scope().reuse_variables()
                self.lemma_outputs_runtime, _ = seq2seq.rnn_decoder(embedded_lemma_characters, self.output,
                                                                    decoder_cell, loop_function=loop)

                lemma_char_logits = []
                for output_train in self.lemma_outputs_train:
                    output_train_activation = tf.matmul(output_train, self.lemma_W) + self.lemma_B
                    lemma_char_logits.append(output_train_activation)

                lemma_char_weights = []
                for lemma_chars in self.lemma_char_inputs[1:]:
                    lemma_char_weights.append(tf.to_float(tf.not_equal(lemma_chars, 0)))

                self.lemmatizer_loss = seq2seq.sequence_loss(lemma_char_logits, self.lemma_char_inputs[1:],
                                                             lemma_char_weights, len(self.alphabet))

                self.cost += tf.reduce_mean(self.lemmatizer_loss)

        self.global_step = tf.Variable(0, trainable=False)
        def decay(learning_rate, exponent, iteration_steps):
            return tf.train.exponential_decay(learning_rate, self.global_step,
                    iteration_steps, exponent, staircase=True)

        self.optimizer = eval('tf.train.' + optimizer_desc)
        self.train = self.optimizer.minimize(self.cost, global_step=self.global_step)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        self.write_summaries = write_summaries
        if write_summaries:
            tf.train.SummaryWriter("logs", self.session.graph_def)


    def learn(self, words, chars, tags, lengths):
        """Learn from the given minibatch."""

        initial_state = np.zeros([words.shape[0], 2 * self.lstm_size])

        fd = {
            self.tags:tags,
            self.sentence_lengths: lengths,
            self.dropout_prob: np.array([0.5]),
            self.forward_initial_state: initial_state,
            self.backward_initial_state: initial_state
        }
        if self.word_embedding_size: fd[self.words] = words
        if self.char_embedding_size: fd[self.characters] = chars

        _, cost = self.session.run([self.train, self.cost], feed_dict=fd)

        return cost


    def predict(self, words, chars, lengths):
        """Predict tags for the given minibatch."""

        initial_state = np.zeros([words.shape[0], 2 * self.lstm_size])

        fd = {
            self.sentence_lengths: lengths,
            self.dropout_prob: np.array([1]),
            self.forward_initial_state: initial_state,
            self.backward_initial_state: initial_state
        }
        if self.word_embedding_size: fd[self.words] = words
        if self.char_embedding_size: fd[self.characters] = chars

        logits = self.session.run(self.logits, feed_dict=fd)

        return np.argmax(logits, axis=2)

    def tag_single_sentence(self, words, chars):
        """Tags a sentence of arbitrary length."""

        initial_state = np.zeros([1, 2 * self.lstm_size])
        backward_initial_state = np.zeros([1, 2 * self.lstm_size])

        tags = []
        for start in xrange(0, len(words), self.num_steps):
            w = np.zeros((1, self.num_steps), dtype='int32')
            for i, w_id in enumerate(words[start:start+self.num_steps]):
                w[0, i] = w_id
            c = np.zeros((1, self.num_steps, self.num_chars), dtype='int32')
            for i, chared_word in enumerate(chars[start:start+self.num_steps]):
                for j, c_id in enumerate(chared_word[:self.num_chars]):
                    c[0, i, j] = c_id

            fd = {
                self.sentence_lengths: [self.num_steps],
                self.dropout_prob: np.array([1]),
                self.forward_initial_state: initial_state,
                self.backward_initial_state: backward_initial_state
            }
            if self.word_embedding_size: fd[self.words] = w
            if self.char_embedding_size: fd[self.characters] = c

            logits, state = self.session.run([self.logits, self.states[-1]], feed_dict=fd)

            initial_state = state
            tags.extend(np.argmax(logits[0], axis=1))

        return [int(x) for x in tags[:len(words)]]
