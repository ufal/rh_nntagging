import math
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
        self.lstm_size = word_embedding_size + 2 * char_embedding_size ###

        self.vocab = vocab
        self.tagset = tagset
        self.alphabet = alphabet

        self.forward_initial_state = tf.placeholder(tf.float32, [None, rnn_cell.BasicLSTMCell(self.lstm_size).state_size])
        self.backward_initial_state = tf.placeholder(tf.float32, [None, rnn_cell.BasicLSTMCell(self.lstm_size).state_size])
        self.sentence_lengths = tf.placeholder(tf.int64, [None])
        self.tags = tf.placeholder(tf.int32, [None, num_steps])
        self.dropout_prob = tf.placeholder(tf.float32, [1])
        self.generate_lemmas = generate_lemmas

        input_list = []

        # Word-level embeddings
        if word_embedding_size:
            self.words = tf.placeholder(tf.int32, [None, num_steps], name='words')
            word_embeddings = tf.Variable(tf.random_uniform([len(vocab), word_embedding_size], -1.0, 1.0))
            we_lookup = tf.nn.embedding_lookup(word_embeddings, self.words)

            input_list.append(we_lookup)

        # Character-level embeddings
        if char_embedding_size:
            self.chars = tf.placeholder(tf.int32, [None, num_steps, num_chars], name='chars')

            char_embeddings = \
                tf.Variable(tf.random_uniform([len(alphabet), char_embedding_size], -1.0, 1.0))
            ce_lookup = tf.nn.embedding_lookup(char_embeddings, self.chars)

            char_inputs = [tf.squeeze(input_, [1]) for input_ in
                           tf.split(1, num_chars, tf.reshape(ce_lookup, [-1, num_chars, char_embedding_size],
                                                             name="reshape-char_inputs"))]

            with tf.variable_scope('char_forward'):
                char_lstm = rnn_cell.BasicLSTMCell(char_embedding_size)
                char_outputs, char_states = rnn.rnn(
                    cell=char_lstm,
                    inputs=char_inputs, dtype=tf.float32)
                    #initial_state=char_lstm.zero_state(-1, tf.float32))

            with tf.variable_scope('char_backward'):
                char_lstm_rev = rnn_cell.BasicLSTMCell(char_embedding_size)
                char_outputs_rev, char_states_rev = rnn.rnn(
                    cell=char_lstm_rev,
                    inputs=list(reversed(char_inputs)), dtype=tf.float32)

            last_char_lstm_state = tf.split(1, 2, char_states[num_chars - 1])[1]
            last_char_lstm_state_rev = tf.split(1, 2, char_states_rev[num_chars - 1])[1]

            last_char_states = \
                tf.reshape(last_char_lstm_state, [-1, num_steps, char_embedding_size],
                           name="reshape-charstates")
            last_char_states_rev = tf.reshape(last_char_lstm_state_rev, [-1, num_steps, char_embedding_size], name="reshape-charstates_rev")

            char_output = tf.concat(2, [last_char_states, last_char_states_rev])

            input_list.append(char_output)

        # All inputs correctly sliced
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, tf.concat(2, input_list))]

        with tf.variable_scope('forward'):
            lstm = rnn_cell.BasicLSTMCell(self.lstm_size)
            outputs, states = rnn.rnn(
                cell=lstm,
                inputs=inputs, dtype=tf.float32,
                initial_state=self.forward_initial_state)#,
                #sequence_length=self.sentence_lengths)

        with tf.variable_scope('backward'):
            lstm_rev = rnn_cell.BasicLSTMCell(self.lstm_size)
            outputs_rev, states_rev = rnn.rnn(
                cell=lstm_rev,
                inputs=list(reversed(inputs)), dtype=tf.float32,
                initial_state=self.backward_initial_state)

        outputs_bidi = [tf.concat(1, [o1, o2]) for o1, o2 in zip(outputs, reversed(outputs_rev))]

        output = tf.reshape(tf.concat(1, outputs_bidi), [-1, 2 * self.lstm_size],
                            name="reshape-outputs_bidi")

        # We are computing only the logits, not the actual softmax -- while
        # computing the loss, it is done by the sequence_loss_by_example and
        # during the runtime classification, the argmax over logits is enough.

        logits_flatten = tf.nn.xw_plus_b(
            output,
            tf.get_variable("softmax_w", [2 * self.lstm_size, len(tagset)]),
            tf.get_variable("softmax_b", [len(tagset)]))

        self.logits = tf.reshape(logits_flatten, [-1, num_steps, len(tagset)], name="reshape-logits")
        self.states = states

        # output maks: compute loss only if it insn't a padded word (i.e. zero index)
        output_mask = tf.reshape(tf.to_float(tf.not_equal(self.tags, 0)), [-1])

        tagging_loss = seq2seq.sequence_loss_by_example(
            logits=[logits_flatten],
            targets=[tf.reshape(self.tags, [-1])],
            weights=[output_mask],
            num_decoder_symbols=len(tagset))

        self.cost = tf.reduce_mean(tagging_loss)

        if generate_lemmas:
            with tf.variable_scope('decoder'):
                self.lemma_chars = tf.placeholder(tf.int32, [None, num_steps, num_chars + 2],
                                                  name='lemma_chars')

                lemma_state_size = self.lstm_size

                lemma_w = tf.Variable(tf.random_uniform([lemma_state_size, len(alphabet)], 0.5),
                                           name="state_to_char_w")
                lemma_b = tf.Variable(tf.fill([len(alphabet)], - math.log(len(alphabet))),
                                           name="state_to_char_b")
                lemma_char_embeddings = tf.Variable(tf.random_uniform([len(alphabet), lemma_state_size], -0.5, 0.5),
                                                    name="char_embeddings")

                lemma_char_inputs = \
                    [tf.squeeze(input_, [1]) for input_ in
                        tf.split(1, num_chars + 2, tf.reshape(self.lemma_chars, [-1, num_chars + 2],
                                                              name="reshape-lemma_char_inputs"))]

                def loop(prev_state, _):
                    # it takes the previous hidden state, finds the character and formats it
                    # as input for the next time step ... used in the decoder in the "real decoding scenario"
                    out_activation = tf.matmul(prev_state, lemma_w) + lemma_b
                    prev_char_index = tf.argmax(out_activation, 1)
                    return tf.nn.embedding_lookup(lemma_char_embeddings, prev_char_index)

                embedded_lemma_characters = []
                for lemma_chars in lemma_char_inputs[:-1]:
                    embedded_lemma_characters.append(tf.nn.embedding_lookup(lemma_char_embeddings, lemma_chars))

                decoder_cell = rnn_cell.BasicLSTMCell(lemma_state_size)
                lemma_outputs_train, _ = seq2seq.rnn_decoder(embedded_lemma_characters, output, decoder_cell)

                tf.get_variable_scope().reuse_variables()
                lemma_outputs_runtime, _ = \
                        seq2seq.rnn_decoder(embedded_lemma_characters, output, decoder_cell,
                                            loop_function=loop)

                lemma_char_logits = []
                for output_train in lemma_outputs_train:
                    output_train_activation = tf.matmul(output_train, lemma_w) + lemma_b
                    lemma_char_logits.append(output_train_activation)

                lemma_char_weights = []
                for lemma_chars in lemma_char_inputs[1:]:
                    lemma_char_weights.append(tf.to_float(tf.not_equal(lemma_chars, 0)))

                lemmatizer_loss = seq2seq.sequence_loss(lemma_char_logits, lemma_char_inputs[1:],
                                                        lemma_char_weights, len(alphabet))

                self.cost += tf.reduce_mean(lemmatizer_loss)

        global_step = tf.Variable(0, trainable=False)
        def decay(learning_rate, exponent, iteration_steps):
            return tf.train.exponential_decay(learning_rate, global_step,
                                              iteration_steps, exponent, staircase=True)

        optimizer = eval('tf.train.' + optimizer_desc)
        self.train = optimizer.minimize(self.cost, global_step=global_step)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        if write_summaries:
            self.summary_writer = tf.train.SummaryWriter("logs", self.session.graph_def)


    def learn(self, words, chars, tags, lengths, lemma_chars):
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
        if self.char_embedding_size: fd[self.chars] = chars
        if self.generate_lemmas: fd[self.lemma_chars] = lemma_chars

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
        if self.char_embedding_size: fd[self.chars] = chars

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
            if self.char_embedding_size: fd[self.chars] = c

            logits, state = self.session.run([self.logits, self.states[-1]], feed_dict=fd)

            initial_state = state
            tags.extend(np.argmax(logits[0], axis=1))

        return [int(x) for x in tags[:len(words)]]
