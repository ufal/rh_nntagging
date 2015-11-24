import chainer
import chainer.optimizers
import chainer.functions as F
import numpy as np
import cPickle
import copy
import logging


class Tagger(object):
    """LSTM tagger model."""
    def __init__(self, vocab, tags, n_lstm_cells=128, model=None):
        """Initialize the tagger model with given vocabulary and tags.

        Args:
            vocab: Vocab-type object that holds the mapping between words and
                    word IDs
            tags: Vocab-type object that holds the mapping between tags and tag
                    IDs
            n_lstm_cells: number of LSTM cells
            model: pre-built model; if specified a new model won't be initialized
        """
        self.n_lstm_cells = 128
        self.vocab = vocab
        self.tags = tags

        if model == None:
            self._create_and_initialize_model(tags, vocab)
        else:
            self.model = model

        self.optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.1, beta2=0.001, eps=1e-8)
        self.optimizer.setup(self.model)

    def _create_and_initialize_model(self, tags, vocab):
        # The model is feed-forward LSTM:
        # (word_id -> word_emb)_t -> LSTM -> (distribution over tag_id)_t

        self.model = chainer.FunctionSet()
        self.model.embed = F.EmbedID(len(vocab), self.n_lstm_cells)
        self.model.lstm_x_to_h = F.Linear(self.n_lstm_cells,
                                          4 * self.n_lstm_cells)
        self.model.lstm_h_to_h = F.Linear(self.n_lstm_cells,
                                          4 * self.n_lstm_cells)
        self.model.lstm2_x_to_h = F.Linear(self.n_lstm_cells,
                                          4 * self.n_lstm_cells)
        self.model.lstm2_h_to_h = F.Linear(self.n_lstm_cells,
                                          4 * self.n_lstm_cells)
        self.model.yclf = F.Linear(self.n_lstm_cells, len(tags))

        # Randomly initialize the parameters.
        for param in self.model.parameters:
            param[:] = np.random.uniform(-0.1, 0.1, param.shape)

    def serialize(self):
        """Return a serialized version of the tagger."""
        return {
            'vocab': self.vocab,
            'tags': self.tags,
            'model': copy.deepcopy(self.model).to_cpu(),
            'n_lstm_cells': self.n_lstm_cells
        }

    @staticmethod
    def deserialize(data):
        """Return an instance of Tagger from its serialized version."""
        obj = Tagger(
            vocab=data['vocab'],
            tags=data['tags'],
            n_lstm_cells=data['n_lstm_cells'],
            model=data['model']
        )

        return obj

    def save(self, fname):
        """Save model to file."""
        logging.debug('Saving model to: %s' % fname)
        logging.info('  Emb matrix first params: %s' % self.model.embed.W.flat[:5])
        with open(fname, 'w') as f_out:
            cPickle.dump(self.serialize(), f_out, -1)

    @staticmethod
    def load(fname):
        """Load model from file and return it."""
        logging.debug('Loading model from: %s' % fname)
        with open(fname) as f_in:
            data = cPickle.load(f_in)
            obj = Tagger.deserialize(data)
            logging.info('  Emb matrix first params: %s' % obj.model.embed.W.flat[:5])
            return obj

    def _compute_seq(self, mb_x, train, lstm_state, reverse=False):
        mb_size = mb_x.shape[0]
        n_steps = mb_x.shape[1]

        H = []

        ndx = range(n_steps)
        if reverse:
            ndx = reversed(ndx)

        for t in ndx:
            x_t = chainer.Variable(mb_x[:, t], volatile=not train)
            c_tm1 = lstm_state['c']
            h_tm1 = lstm_state['h']

            e_t = self.model.embed(x_t)

            c_t, h_t = F.lstm(c_tm1, self.model.lstm_x_to_h(e_t) + self.model.lstm_h_to_h(h_tm1))

            lstm_state['c'] = c_t
            lstm_state['h'] = h_t

            H.append(h_t)

        if reverse:
            H = H[::-1]

        return H

    def forward(self, mb_x, mb_y, train=True):
        """Run the model on given minibatch.

        Args:
            mb_x: numpy.array of float32, dimensions: (minibatch, time, data)
                      specify input data mb_x[3, 5, 9] is a float value of
                      9th dimension of the input vector, of 3rd example in minibatch
                      at 5th timestep in the sequence
            mb_y: numpy.array of ints, dimensions: (minibatch, time) specifying
                      correct label; -1 means that this example will be ignored
                      and does not influence the loss
            train: bool, Is the result of this call going to be used for training?

        Returns: loss as Chainer variable and predictions as numpy array
        """
        mb_size = mb_x.shape[0]
        n_steps = mb_x.shape[1]

        loss = 0.0

        lstm_state = self.create_lstm_initial_state(batchsize=mb_size)
        lstm_state2 = self.create_lstm_initial_state(batchsize=mb_size)

        H_fwd = self._compute_seq(mb_x, train, lstm_state)
        H_bwd = self._compute_seq(mb_x, train, lstm_state2, reverse=True)

        y_hat = []
        for t in range(n_steps):

            y_t = chainer.Variable(mb_y[:, t], volatile=not train)

            yhat_t = self.model.yclf(H_fwd[t] + H_bwd[t])

            l_t = F.softmax_cross_entropy(yhat_t, y_t)
            y_hat.append(l_t.creator.y)

            loss += l_t * 1.0 / (n_steps * mb_size)

        y_hat = np.array(y_hat).swapaxes(0, 1)

        return loss, y_hat

    def create_lstm_initial_state(self, batchsize, train=True):
        """Build initial hidden and cell state for LSTM."""
        return {name: chainer.Variable(np.zeros((batchsize, self.n_lstm_cells),
                                                dtype=np.float32),
                                       volatile=not train)
                for name in ('c', 'h',)}

    def learn(self, mb_x, mb_y):
        """Learn from the given minibatch."""
        self.optimizer.zero_grads()

        loss, y_hat = self.forward(mb_x, mb_y, train=True)

        loss.backward()

        self.optimizer.update()

        return loss.data

    def predict(self, mb_x):
        """Predict tags for the given minibatch."""
        _, y_hat = self.forward(mb_x, np.zeros(mb_x.shape, dtype='int32'))

        return np.argmax(y_hat, axis=2)
