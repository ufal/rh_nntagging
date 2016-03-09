import math
import numpy as np
import tensorflow as tf
import datetime

from tensorflow.python.ops import array_ops
from tensorflow.models.rnn import rnn_cell, rnn, seq2seq


    
def _reverse_seq(input_seq, lengths):
    """Reverse a list of Tensors up to specified lengths.
    
      Args:
    input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
    lengths:   A tensor of dimension batch_size, containing lengths for each
    sequence in the batch. If "None" is specified, simply reverses
    the list.
    
      Returns:
    time-reversed sequence
    """
    if lengths is None:
        return list(reversed(input_seq))
        
    for input_ in input_seq:
        input_.set_shape(input_.get_shape().with_rank(2))
          
    # Join into (time, batch_size, depth)
    s_joined = array_ops.pack(input_seq)
          
    # Reverse along dimension 0
    s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
    # Split again into list
    result = array_ops.unpack(s_reversed)
    return result



class CharacterEmbeddingEncoder(object):
    """
    Bidirectional LSTM character-level word embedding encoder for sentence-level operations

    INPUTS:
        list of characters (batch x num_steps x num_chars)
        list of word lengths (batch x num_steps)

    OUTPUTS:
        flatten vector of character embeddings ((batch * num_steps) x num_chars x char_embedding_size)
        combined last state of the bidirectional LSTM (batch x num_steps x char_embedding_size * 2)
    """
        
    def __init__(self, alphabet, char_embedding_size, num_chars, num_steps):
        """
        Arguments:
            alphabet: Vocabulary of possible characters
            char_embedding_size: Size of the resulting character embedding
            num_chars: Maximum word length
            num_steps: Maximum sentence length
        """

        ## TODO rozhodnout, jestli umoznit rozdilny dylky pro embedding jednoho pismene a celyho slova
        ## delalo by se to pres input_size argument BasicLSTMCell konstruktoru
        
        self.chars = tf.placeholder(tf.int32, [None, num_steps, num_chars], name='chars')
        self.word_lengths = tf.placeholder(tf.int64, [None, num_steps], name='word_lengths')
        word_lengths_flat = tf.reshape(self.word_lengths, [-1])

        # alphabet x char_embedding_size
        char_embeddings = tf.Variable(tf.random_uniform([len(alphabet), char_embedding_size], -1.0, 1.0))

        # shape(self.chars) + shape(char_embeddings)[1:]
        # batch x num_steps x num_chars x char_embedding_size
        lookup = tf.nn.embedding_lookup(char_embeddings, self.chars)

        # (batch * num_steps) x num_chars x char_embedding_size
        lookup_reshaped = tf.reshape(lookup, [-1, num_chars, char_embedding_size], name="reshape-char-lookup")

        # [list indexovany podle num_chars] (batch * num_steps) x char_embedding_size
        char_inputs = [ tf.squeeze(input_, [1]) for input_ in tf.split(1, num_chars, lookup_reshaped) ]

        with tf.variable_scope('char_forward'):
            forward_lstm = rnn_cell.BasicLSTMCell(char_embedding_size)
            _, forward_state = rnn.rnn(
                cell=forward_lstm,
                inputs=char_inputs,
                sequence_length=word_lengths_flat,
                dtype=tf.float32)

            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('char_backward'):
            backward_lstm = rnn_cell.BasicLSTMCell(char_embedding_size)
            _, backward_state = rnn.rnn(
                cell=backward_lstm,
                inputs=_reverse_seq(char_inputs, word_lengths_flat),
                sequence_length=word_lengths_flat,
                dtype=tf.float32)

            tf.get_variable_scope().reuse_variables()

        # NOTE stav je rozdelenej na dve pulky.
        # - prvni je c (hodnota co zustava uvnitr lstm)
        # - druha je h (tanh(c) + sigmoid(o), tj. hodnota, ktera se posila na output)
        # TODO nechceme tedy brat spis prvni polovinu?
            
        # (batch * num_steps) x char_embedding_size
        forward_states_flat = tf.split(1, 2, forward_state)[1]
        backward_states_flat = tf.split(1, 2, backward_state)[1]

        # batch x num_steps x char_embedding_size
        forward_states = tf.reshape(forward_states_flat, [-1, num_steps, char_embedding_size])
        backward_states = tf.reshape(backward_states_flat, [-1, num_steps, char_embedding_size])

        # batch x num_steps x (char_embedding_size * 2)
        self.outputs = tf.concat(2, [forward_states, backward_states])
        self.lookup_reshaped = lookup_reshaped



class BidirectionalLSTM(object):
    """
    Bidirectional LSTM model for tagging

    INPUTS:
        List of inputs. Each element in the list has to have shape (batch x num_steps x <input>)

    OUTPUTS:
        output from bidirectional lstm ((batch * num_steps) x (2 * lstm_size))
        flatten logits ((batch * num_steps) x output_size)
        argmax over flatten logits ((batch * num_steps) x output_size)
        reshaped logits (batch x num_steps x output_size)
        last LSTM forward state (batch x (2 * lstm_size))
        
    """

    
    def __init__(self, lstm_size, input_list, num_steps,
                 forward_initial_state, backward_initial_state, sentence_lengths,
                 dropout_out, output_size):
        """
        Arguments:
            lstm_size: size of lstm output vector
            input_list: list of input embeddings (word, char)
            num_steps: maximum length of a sentence        
        """

        # [list indexovany podle num_steps] batch x (word_embedding_size + 2 * char_embedding_size)
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, tf.concat(2, input_list))]

        with tf.variable_scope('forward'):
            forward_lstm = rnn_cell.BasicLSTMCell(lstm_size)
            forward_outputs, forward_state = rnn.rnn(
                cell=forward_lstm,
                inputs=inputs,
                initial_state=forward_initial_state,
                sequence_length=sentence_lengths,
                dtype=tf.float32)

            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('backward'):
            backward_lstm = rnn_cell.BasicLSTMCell(lstm_size)
            backward_outputs_reversed, backward_state = rnn.rnn(
                cell=backward_lstm,
                inputs=_reverse_seq(inputs, sentence_lengths),
                initial_state=backward_initial_state,
                sequence_length=sentence_lengths)

            backward_outputs = _reverse_seq(backward_outputs_reversed, sentence_lengths)
            tf.get_variable_scope().reuse_variables()

        # [list indexovany podle num_steps] batch x (2 * lstm_size)
        bidi_outputs = [tf.concat(1, [o1, o2]) for o1, o2 in zip(forward_outputs, reversed(backward_outputs))]

        # (batch * num_steps) x (2 * lstm_size)
        lstm_output = tf.reshape(tf.concat(1, bidi_outputs), [-1, 2 * lstm_size])
        self.lstm_output = tf.nn.dropout(lstm_output, dropout_out)

        # (batch * num_steps) x output_size
        self.logits_flat = tf.nn.xw_plus_b(
            lstm_output,
            tf.get_variable('softmax_w', [2 * lstm_size, output_size]),
            tf.get_variable('softmax_b', [output_size]))
        
        self.argmax = tf.to_int32(tf.argmax(self.logits_flat, dimension=1))
        
        # batch x num_steps x output_size
        self.logits = tf.reshape(self.logits_flat, [-1, num_steps, output_size])

        self.last_state = forward_state
        
        


class Tagger(object):
    """LSTM tagger model."""
    def __init__(self, vocab, tagset, alphabet, word_embedding_size,
                 char_embedding_size, num_chars, num_steps, optimizer_desc,
                 generate_lemmas, l2, dropout_prob_values, experiment_name,
                 supply_form_characters_to_lemma, threads=0, seed=None, write_summaries=True, use_attention=True, scheduled_sampling=None):
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

        self.dropout_prob_values = dropout_prob_values

        self.forward_initial_state = tf.placeholder(tf.float32, [None, rnn_cell.BasicLSTMCell(self.lstm_size).state_size], name="forward_lstm_initial_state")
        self.backward_initial_state = tf.placeholder(tf.float32, [None, rnn_cell.BasicLSTMCell(self.lstm_size).state_size], name="backward_lstm_initial_state")
        self.sentence_lengths = tf.placeholder(tf.int64, [None], name="sentence_lengths")
        self.tags = tf.placeholder(tf.int32, [None, num_steps], name="ground_truth_tags")
        self.dropout_prob = tf.placeholder(tf.float32, [None], name="dropout_keep_p")
        self.generate_lemmas = generate_lemmas

        global_step = tf.Variable(0, trainable=False)

        input_list = []
        regularize = []

        # Word-level embeddings
        if word_embedding_size:
            self.words = tf.placeholder(tf.int32, [None, num_steps], name='words')
            word_embeddings = tf.Variable(tf.random_uniform([len(vocab), word_embedding_size], -1.0, 1.0))
            we_lookup = tf.nn.embedding_lookup(word_embeddings, self.words)

            input_list.append(we_lookup)

        # Character-level embeddings
        if char_embedding_size:
            self.char_encoder = CharacterEmbeddingEncoder(
                self.alphabet,
                self.char_embedding_size,
                self.num_chars,
                self.num_steps)
            
            input_list.append(self.char_encoder.outputs)

        # All inputs correctly sliced
        input_list_dropped = [tf.nn.dropout(x, self.dropout_prob[0]) for x in input_list]

        self.tagging_lstm = BidirectionalLSTM(
            self.lstm_size,
            input_list_dropped,
            self.num_steps,
            self.forward_initial_state,
            self.backward_initial_state,
            self.sentence_lengths,
            self.dropout_prob[1],
            len(self.tagset))
        

        # output maks: compute loss only if it insn't a padded word (i.e. zero index)
        output_mask = tf.reshape(tf.to_float(tf.not_equal(self.tags, 0)), [-1])
        gt_tags_flat = tf.reshape(self.tags, [-1])

        tagging_loss = seq2seq.sequence_loss_by_example(
            logits=[self.tagging_lstm.logits_flat],
            targets=[gt_tags_flat],
            weights=[output_mask])

        tagging_accuracy = \
            tf.reduce_sum(tf.to_float(tf.equal(self.tagging_lstm.argmax, gt_tags_flat)) * output_mask) \
                / tf.reduce_sum(output_mask)
        tf.scalar_summary('train_accuracy', tagging_accuracy, collections=["train"])
        tf.scalar_summary('dev_accuracy', tagging_accuracy, collections=["dev"])

        self.cost = tf.reduce_mean(tagging_loss)

        tf.scalar_summary('train_tagging_loss', tf.reduce_mean(tagging_loss), collections=["train"])
        tf.scalar_summary('dev_tagging_loss', tf.reduce_mean(tagging_loss), collections=["dev"])

        if generate_lemmas:
            with tf.variable_scope('decoder'):
                self.lemma_chars = tf.placeholder(tf.int32, [None, num_steps, num_chars + 2],
                                                  name='lemma_chars') 

                lemma_state_size = self.lstm_size

                lemma_w = tf.Variable(tf.random_uniform([lemma_state_size, len(alphabet)], 0.5),
                                           name="state_to_char_w")
                lemma_b = tf.Variable(tf.fill([len(alphabet)], - math.log(len(alphabet))),
                                           name="state_to_char_b")
                lemma_char_embeddings = tf.Variable(tf.random_uniform([len(alphabet), lemma_state_size / (2 if supply_form_characters_to_lemma else 1)], -0.5, 0.5),
                                                    name="char_embeddings")

                lemma_char_inputs = \
                    [tf.squeeze(input_, [1]) for input_ in
                        tf.split(1, num_chars + 2, tf.reshape(self.lemma_chars, [-1, num_chars + 2],
                                                              name="reshape-lemma_char_inputs"))]

                if supply_form_characters_to_lemma:
                    char_inputs_zeros = \
                        [tf.squeeze(chars, [1]) for chars in
                            tf.split(1, num_chars, tf.reshape(self.char_encoder.chars, [-1, num_chars],
                                                              name="reshape-char_inputs_zeros"))]
                    char_inputs_zeros.append(char_inputs_zeros[0] * 0)

                    def loop(prev_state, i):
                        # it takes the previous hidden state, finds the character and formats it
                        # as input for the next time step ... used in the decoder in the "real decoding scenario"
                        out_activation = tf.matmul(prev_state, lemma_w) + lemma_b
                        prev_char_index = tf.argmax(out_activation, 1)
                        return tf.concat(1, [tf.nn.embedding_lookup(lemma_char_embeddings, prev_char_index),
                                             tf.nn.embedding_lookup(lemma_char_embeddings, char_inputs_zeros[i])])

                    embedded_lemma_characters = []
                    for lemma_chars, form_chars in zip(lemma_char_inputs[:-1], char_inputs_zeros):
                        embedded_lemma_characters.append(
                            tf.concat(1, [tf.nn.embedding_lookup(lemma_char_embeddings, lemma_chars),
                                          tf.nn.embedding_lookup(lemma_char_embeddings, form_chars)])
                        )
                else:
                    def loop(prev_state, _):
                        # it takes the previous hidden state, finds the character and formats it
                        # as input for the next time step ... used in the decoder in the "real decoding scenario"
                        out_activation = tf.matmul(prev_state, lemma_w) + lemma_b
                        prev_char_index = tf.argmax(out_activation, 1)
                        return tf.nn.embedding_lookup(lemma_char_embeddings, prev_char_index)

                    embedded_lemma_characters = []
                    for lemma_chars in lemma_char_inputs[:-1]:
                        embedded_lemma_characters.append(tf.nn.embedding_lookup(lemma_char_embeddings, lemma_chars))


                def sampling_loop(prev_state, i):
                    threshold = scheduled_sampling / (scheduled_sampling + tf.exp(tf.to_float(global_step)))
                    condition = tf.less_equal(tf.random_uniform(tf.shape(embedded_lemma_characters[0])), threshold)
                    return tf.select(condition, embedded_lemma_characters[i], loop(prev_state,i))

                decoder_cell = rnn_cell.BasicLSTMCell(lemma_state_size)

                if scheduled_sampling:
                    lf = sampling_loop
                else:
                    lf = None


                if use_attention:
                    lemma_outputs_train, _ = seq2seq.attention_decoder(embedded_lemma_characters, self.tagging_lstm.lstm_output, self.char_encoder.lookup_reshaped, decoder_cell, loop_function=lf)
                else:
                    lemma_outputs_train, _ = seq2seq.rnn_decoder(embedded_lemma_characters, self.tagging_lstm.lstm_output, decoder_cell, loop_function=lf)


                tf.get_variable_scope().reuse_variables()
                #regularize.append(tf.get_variable('attention_decoder/BasicLSTMCell/Linear/Matrix'))

                tf.get_variable_scope().reuse_variables()

                if use_attention:
                    lemma_outputs_runtime, _ = \
                        seq2seq.attention_decoder(embedded_lemma_characters, self.tagging_lstm.lstm_output, self.char_encoder.lookup_reshaped, decoder_cell,
                            loop_function=loop)
                else:
                    lemma_outputs_runtime, _ = \
                        seq2seq.rnn_decoder(embedded_lemma_characters, self.tagging_lstm.lstm_output, decoder_cell,
                            loop_function=loop)

                lemma_char_logits_train = \
                    [tf.matmul(o, lemma_w) + lemma_b for o in lemma_outputs_train]

                lemma_char_logits_runtime = \
                    [tf.matmul(o, lemma_w) + lemma_b for o in lemma_outputs_runtime]

                self.lemmas_decoded = \
                    tf.reshape(tf.transpose(tf.argmax(tf.pack(lemma_char_logits_runtime), 2)), [-1, num_steps, num_chars + 1])

                lemma_char_weights = []
                for lemma_chars in lemma_char_inputs[1:]:
                    lemma_char_weights.append(tf.to_float(tf.not_equal(lemma_chars, 0)))

                lemmatizer_loss = seq2seq.sequence_loss(lemma_char_logits_train, lemma_char_inputs[1:],
                                                        lemma_char_weights)

                lemmatizer_loss_runtime = \
                        seq2seq.sequence_loss(lemma_char_logits_runtime, lemma_char_inputs[1:],
                                              lemma_char_weights)

                tf.scalar_summary('train_lemma_loss_with_gt_inputs',
                                  tf.reduce_mean(lemmatizer_loss), collections=["train"])
                tf.scalar_summary('dev_lemma_loss_with_gt_inputs',
                                  tf.reduce_mean(lemmatizer_loss), collections=["dev"])

                tf.scalar_summary('train_lemma_loss_with_decoded_inputs',
                                  tf.reduce_mean(lemmatizer_loss_runtime), collections=["train"])
                tf.scalar_summary('dev_lemma_loss_with_decoded_inputs',
                                  tf.reduce_mean(lemmatizer_loss_runtime), collections=["dev"])

                self.cost += tf.reduce_mean(lemmatizer_loss) + tf.reduce_mean(lemmatizer_loss_runtime)

        self.cost += l2 * sum([tf.nn.l2_loss(variable) for variable in regularize])

        tf.scalar_summary('train_optimization_cost', self.cost, collections=["train"])
        tf.scalar_summary('dev_optimization_cost', self.cost, collections=["dev"])

        def decay(learning_rate, exponent, iteration_steps):
            return tf.train.exponential_decay(learning_rate, global_step,
                                              iteration_steps, exponent, staircase=True)

        optimizer = eval('tf.train.' + optimizer_desc)
        self.train = optimizer.minimize(self.cost, global_step=global_step)

        if threads > 0:
            self.session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                            intra_op_parallelism_threads=threads))
        else:
            self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        if write_summaries:
            self.summary_train = tf.merge_summary(tf.get_collection("train"))
            self.summary_dev = tf.merge_summary(tf.get_collection("dev"))
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            self.summary_writer = tf.train.SummaryWriter("logs/"+timestamp+"_"+experiment_name)

        self.saver = tf.train.Saver(tf.trainable_variables())
        self.steps = 0


    def learn(self, words, chars, tags, lengths, lemma_chars, chars_lengths):
        """Learn from the given minibatch."""

        initial_state = np.zeros([words.shape[0], 2 * self.lstm_size])
        self.steps += 1

        fd = {
            self.tags:tags,
            self.sentence_lengths: lengths,
            self.dropout_prob: self.dropout_prob_values,
            self.forward_initial_state: initial_state,
            self.backward_initial_state: initial_state
        }
        if self.word_embedding_size: fd[self.words] = words
        if self.char_embedding_size:
            fd[self.char_encoder.chars] = chars
            fd[self.char_encoder.word_lengths] = chars_lengths
        if self.generate_lemmas: fd[self.lemma_chars] = lemma_chars

        _, cost, summary_str = \
                self.session.run([self.train, self.cost, self.summary_train], feed_dict=fd)
        if self.steps % 10 == 0:
            self.summary_writer.add_summary(summary_str, self.steps)

        return cost


    def predict_and_eval(self, words, chars, lengths, tags, lemma_chars, chars_lengths, out_summaries=True):
        """Predict tags for the given minibatch."""

        initial_state = np.zeros([words.shape[0], 2 * self.lstm_size])

        fd = {
            self.tags:tags,
            self.sentence_lengths: lengths,
            self.dropout_prob: np.array([1, 1]),
            self.forward_initial_state: initial_state,
            self.backward_initial_state: initial_state
        }
        if self.word_embedding_size: fd[self.words] = words
        if self.char_embedding_size:
            fd[self.char_encoder.chars] = chars
            fd[self.char_encoder.word_lengths] = chars_lengths
        if self.generate_lemmas: fd[self.lemma_chars] = lemma_chars

        if self.generate_lemmas:
            logits, lemmas, summary_str = \
                    self.session.run([self.tagging_lstm.logits, self.lemmas_decoded, self.summary_dev], feed_dict=fd)
        else:
            logits, summary_str = \
                    self.session.run([self.tagging_lstm.logits, self.summary_dev], feed_dict=fd)
            lemmas = None
        if out_summaries:
            self.summary_writer.add_summary(summary_str, self.steps)

        return np.argmax(logits, axis=2), lemmas

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
            lens = np.zeros((1, self.num_steps), dtype='int32')
            for i, chared_word in enumerate(chars[start:start+self.num_steps]):
                lens[0,i] = len(chared_word[:self.num_chars])
                for j, c_id in enumerate(chared_word[:self.num_chars]):
                    c[0, i, j] = c_id

            fd = {
                self.sentence_lengths: [self.num_steps],
                self.dropout_prob: np.array([1, 1]),
                self.forward_initial_state: initial_state,
                self.backward_initial_state: backward_initial_state
            }
            if self.word_embedding_size: fd[self.words] = w
            if self.char_embedding_size:
                fd[self.char_encoder.chars] = c
                fd[self.char_encoder.word_lengths] = lens

            logits, state = self.session.run([self.tagging_lstm.logits, self.tagging_lstm.last_state], feed_dict=fd)

            initial_state = state
            tags.extend(np.argmax(logits[0], axis=1))

        return [int(x) for x in tags[:len(words)]]


def tagger_from_args(vocab, tags, alphabet, args):
    return Tagger(vocab, tags, alphabet,
                    word_embedding_size=args.word_embedding_size,
                    char_embedding_size=args.char_embedding_size,
                    num_chars=args.max_word_length,
                    num_steps=args.max_sentence_length,
                    optimizer_desc=args.optimizer,
                    generate_lemmas=args.generate_lemmas,
                    l2=args.l2,
                    dropout_prob_values=[float(x) for x in args.dropout.split(",")],
                    experiment_name=args.exp_name,
                    supply_form_characters_to_lemma=args.supply_form_characters_to_lemma,
                    threads=args.threads,
                    use_attention=args.use_attention,
                    scheduled_sampling=args.scheduled_sampling)
