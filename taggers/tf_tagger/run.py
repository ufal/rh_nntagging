from collections import deque
import random
import time
import numpy as np
import os
import logging
import tensorflow as tf

import conllu
from tagger import Tagger
from vocab import Vocab
from tagging_dataset import TaggingDataset


class TrainingManager(object):
    def __init__(self, n_train_batches, eval_interval, training_dir=None,
                 tagger_taste_fn=None, tagger_dev_eval_fn=None, tagger_save_fn=None):
        """Takes care of monitoring and managing the model training.

        Args:
            n_train_batches: number of batches in training data
            eval_interval: evaluate model on dev data every `eval_interval` batches
            training_dir: None or a path to directory where models and other
                    logs will be saved
            tagger_taste_fn: a function that is called with stats to print out
                    some example of how the tagger currently works on some data
            tagger_dev_eval_fn: a function that evaluates the model on dev data
                    and returns the models' accuracy
            tagger_save_fn: a function that is called after each evaluation to
                    save the model to a file that is passed as an argument
        """
        self.n_train_batches = n_train_batches
        self.last_eval = 0
        self.eval_interval = eval_interval
        self.tagger_taste_fn = tagger_taste_fn
        self.tagger_dev_eval_fn = tagger_dev_eval_fn
        self.tagger_save_fn = tagger_save_fn

        self.recent_losses = deque(maxlen=n_train_batches)
        self.mb_done = 0
        self.not_improved_by_eps_for = 0
        self.eps = 1e-3
        self.max_dev_perf = 0.0
        self.evals_done = 0
        self.training_dir = training_dir

    def should_continue(self, min_mb_done=None, max_dev_not_improved=None):
        """Shall the training continue?

        Args:
            min_mb_done: How many minibatches need to be seen before the training
                     can terminate?
            max_dev_not_improved: How many times the performance on dev data can
                     NOT improve in a row and we still continue training?

        Returns: bool, whether the training should continue
        """
        if min_mb_done is None: min_mb_done = self.n_train_batches
        if max_dev_not_improved is None: max_dev_not_improved = self.n_train_batches

        if self.mb_done < min_mb_done:
            return True
        else:
            if self.not_improved_by_eps_for > max_dev_not_improved:
                return False
            else:
                return True

    def tick(self, mb_loss, force_eval=False):
        """Report minibatch loss once upon a time.

        Args:
            mb_loss: float, loss of current minibatch
        """
        self.recent_losses.append(mb_loss)
        self.mb_done += 1

        if self.mb_done - self.last_eval >= self.eval_interval or force_eval:
            self.eval_on_dev()
            self.print_stats()
            if self.mb_done - self.last_eval >= self.eval_interval: self.last_eval = self.mb_done


    def print_stats(self):
        """Print current stats of the training."""
        logging.debug('Epoch %d, batches %d/%d, avg_loss(%.8f) max_dev_acc(%.5f) curr_dev_acc(%.5f)'
            % (1 + self.mb_done / self.n_train_batches, self.mb_done % self.n_train_batches,
               self.n_train_batches, np.mean(self.recent_losses), self.max_dev_perf, self.curr_dev_perf)
        )

        self.tagger_taste_fn()

    def eval_on_dev(self):
        """Evaluate the model on dev data."""
        self.evals_done += 1
        tagger_perf = self.tagger_dev_eval_fn()

        if tagger_perf < self.max_dev_perf - self.eps:
            self.not_improved_by_eps_for += 1
        else:
            self.not_improved_by_eps_for = 0

        self.max_dev_perf = max(self.max_dev_perf, tagger_perf)
        self.curr_dev_perf = tagger_perf

        if self.training_dir:
            if not os.path.exists(self.training_dir):
                os.makedirs(self.training_dir)

            self.tagger_save_fn(os.path.join(self.training_dir, 'model_%d.pickle' % self.evals_done))


def taste_tagger(tagger, batches):
    """Print out first 3 examples of the first batch tagged by the model."""
    mb_x, chars, mb_y, lengths = batches[0]
    mb_y_hat = tagger.predict(mb_x, chars, lengths)

    for x, y, y_hat in zip(mb_x, mb_y, mb_y_hat)[:3]:
        line = ' '
        for x_t, y_t, y_hat_t in zip(x, y, y_hat)[:5]:
            line += ' %s(%s/%s)' % (tagger.vocab.rev(x_t), tagger.tagset.rev(y_hat_t), tagger.tagset.rev(y_t))
        logging.debug(line)


def eval_tagger(tagger, batches):
    """Evaluate the tagger on the given data."""
    acc_total = 0.0
    acc_cnt = 0
    for mb_x, chars, mb_y, lengths in batches:
        mb_y_hat = tagger.predict(mb_x, chars, lengths)

        acc_total += compute_accuracy(mb_y, mb_y_hat)
        acc_cnt += 1

    return acc_total / acc_cnt


def compute_accuracy(mb_y, mb_y_hat):
    """Compaute accuracy of classification given the predicted and true labels."""
    eq = mb_y == mb_y_hat
    n_correct = eq[mb_y != 0].sum()
    n_total = (mb_y != 0).sum()

    return n_correct * 1.0 / n_total


def main(args):
    logging.debug('Initializing random seed to 0.')
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)

    logging.debug('Loading training dataset from: %s' % args.training_file)
    train_data = TaggingDataset.load_from_file(args.training_file)
    dev_data = TaggingDataset.load_from_file(None, vocab=train_data.vocab,
            alphabet=train_data.alphabet, tags=train_data.tags)
    logging.debug('Initializing model.')
    tagger = Tagger(train_data.vocab, train_data.tags, train_data.alphabet,
            word_embedding_size=args.word_embedding_size,
            char_embedding_size=args.char_embedding_size,
            num_chars=args.max_word_length,
            num_steps=args.max_sentence_length)

    if not args.skip_train:
        batches_train = train_data.prepare_batches(
                args.batch_size, args.max_sentence_length, args.max_word_length)
        batches_dev = dev_data.prepare_batches(
                args.batch_size, args.max_sentence_length, args.max_word_length)

        train_mgr = TrainingManager(
            len(batches_train), args.eval_interval,
            training_dir=args.training_dir,
            tagger_taste_fn=lambda: taste_tagger(tagger, batches_train),
            tagger_dev_eval_fn=lambda: eval_tagger(tagger, batches_dev),
            tagger_save_fn=lambda fname: tagger.save(fname)
        )

        import signal
        force_eval = {"value": False}
        def handle_sigquit(signal, frame):
            logging.debug("Ctrl+\ recieved, evaluation will be forced.")
            force_eval["value"] = True
            pass
        signal.signal(signal.SIGQUIT, handle_sigquit)

        logging.debug('Starting training.')
        try:
            permuted_batches = []
            while train_mgr.should_continue():
                if not permuted_batches:
                    permuted_batches = batches_train[:]
                    random.shuffle(permuted_batches)
                words, chars, tags, lengths = permuted_batches.pop()
                oov_mask = np.vectorize(lambda x: train_data.vocab.count(x) == 1 and np.random.uniform() < args.oov_sampling_p)(words)
                words = np.where(oov_mask, np.zeros(words.shape), words)
                mb_loss = tagger.learn(words, chars, tags, lengths)

                train_mgr.tick(mb_loss=mb_loss, force_eval=force_eval["value"])
                force_eval["value"] = False
        except KeyboardInterrupt:
            logging.debug("Ctrl+C recieved, stopping training.")

    run_tagger_and_writeout(tagger, dev_data)


def run_tagger_and_writeout(tagger, dev_data):
    logging.debug("Tagging testing data with the trained tagger.")
    for words, chars, _ in dev_data.seqs:

        y_hat = tagger.tag_single_sentence(words, chars)
        y_hat_str = [tagger.tagset.rev(tag_id) for tag_id in y_hat]

        for i, (word, utag) in enumerate(zip(words, y_hat_str)):
            print "{}\t{}\t_\t{}\t_\t_\t_\t_\t_\t_".format(i + 1, dev_data.vocab.rev(word), utag)
        print ""
    logging.debug("Testing data tagged.")


if __name__ == '__main__':
    import utils
    utils.init_logging('Tagger')

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('training_file',
                        help='Training file.')
    parser.add_argument('--training-dir',
                        help='Training directory where logs and models will be saved.')
    parser.add_argument('--load-model',
                        help='Model filename.')
    parser.add_argument('--skip-train', action='store_true', default=False,
                        help='Shall we skip training altogether and just use '+
                             'a stored model for tagging?')
    parser.add_argument('--batch-size', default=50, type=int,
                        help='Batch size.')
    parser.add_argument('--eval-interval', default=100, type=int,
                        help='Evaluate tagger every specified number of batches.')
    parser.add_argument('--oov-sampling-p', default=0.0, type=float,
                        help='Probablity of a word of frequency 1 to be sampled as an OOV.')
    parser.add_argument('--word-embedding-size', default=128, type=int,
                        help='Dimension of word-level word embedding.')
    parser.add_argument('--char-embedding-size', default=16, type=int,
                        help='Dimension of character-level word embedding.')
    parser.add_argument('--max-sentence-length', default=50, type=int,
                        help='Maximum sentence length during training.')
    parser.add_argument('--max-word-length', default=20, type=int,
                        help='Maximum word length during training.')
    parser.add_argument("--optimizer", default=None, type=str,
                        help="Optimizer specification to be specified.")
    args = parser.parse_args()

    main(args)
