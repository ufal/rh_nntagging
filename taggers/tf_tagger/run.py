from collections import deque
import random
import time
import numpy as np
import os
import logging

import conllu
from tagger import Tagger
from vocab import Vocab
from tagging_dataset import TaggingDataset


class TrainingManager(object):
    def __init__(self, print_interval=1.0, eval_interval=10.0, avg_n_losses=100,
                 training_dir=None,
                 tagger_taste_fn=None, tagger_dev_eval_fn=None, tagger_save_fn=None):
        """Takes care of monitoring and managing the model training.

        Args:
            print_interval: print stats every `print_interval` secs
            eval_interval: evaluate model on dev data every `eval_interval` secs
            avg_n_losses: from how many minibatch losses should the final loss
                    be computed?
            training_dir: None or a path to directory where models and other
                    logs will be saved
            tagger_taste_fn: a function that is called with stats to print out
                    some example of how the tagger currently works on some data
            tagger_dev_eval_fn: a function that evaluates the model on dev data
                    and returns the models' accuracy
            tagger_save_fn: a function that is called after each evaluation to
                    save the model to a file that is passed as an argument
        """
        self.last_print = 0
        self.last_eval = 0
        self.print_interval = print_interval
        self.eval_interval = eval_interval
        self.tagger_taste_fn = tagger_taste_fn
        self.tagger_dev_eval_fn = tagger_dev_eval_fn
        self.tagger_save_fn = tagger_save_fn

        self.recent_losses = deque(maxlen=avg_n_losses)
        self.mb_done = 0
        self.not_improved_by_eps_for = 0
        self.eps = 1e-3
        self.max_dev_perf = 0.0
        self.evals_done = 0
        self.training_dir = training_dir

    def should_continue(self, min_mb_done=50, max_dev_not_improved=3):
        """Shall the training continue?

        Args:
            min_mb_done: How many minibatches need to be seen before the training
                     can terminate?
            max_dev_not_improved: How many times the performance on dev data can
                     NOT improve in a row and we still continue training?

        Returns: bool, whether the training should continue
        """
        if self.mb_done < min_mb_done:
            return True
        else:
            if self.not_improved_by_eps_for > max_dev_not_improved:
                return False
            else:
                return True

    def tick(self, mb_loss):
        """Report minibatch loss.

        Args:
            mb_loss: float, loss of current minibatch
        """
        self.recent_losses.append(mb_loss)
        self.mb_done += 1

        if time.time() - self.last_eval > self.eval_interval:
            self.eval_on_dev()
            self.last_eval = time.time()

        if time.time() - self.last_print > self.print_interval:
            self.print_stats()
            self.last_print = time.time()

    def print_stats(self):
        """Print current stats of the training."""
        logging.debug(
            '  mb_done(%d) avg_loss(%.4f) max_dev_acc(%.2f) curr_dev_acc(%.2f)'
            % (self.mb_done, np.mean(self.recent_losses), self.max_dev_perf, self.curr_dev_perf)
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
    mb_x, mb_y = batches[0]
    mb_y_hat = tagger.predict(mb_x)

    logging.debug("Taste: word tag (true tag)")
    for x, y, y_hat in zip(mb_x, mb_y, mb_y_hat)[:3]:
        for x_t, y_t, y_hat_t in zip(x, y, y_hat)[:5]:
            logging.debug(
                "  %10s %10s (%s)"
                % (
                    tagger.vocab.rev(x_t),
                    tagger.tagset.rev(y_hat_t),
                    tagger.tagset.rev(y_t)
                ))
        logging.debug("")


def eval_tagger(tagger, batches):
    """Evaluate the tagger on the given data."""
    acc_total = 0.0
    acc_cnt = 0
    for mb_x, mb_y in batches:
        mb_y_hat = tagger.predict(mb_x)

        acc_total += compute_accuracy(mb_y, mb_y_hat)
        acc_cnt += 1

    return acc_total / acc_cnt


def compute_accuracy(mb_y, mb_y_hat):
    """Compaute accuracy of classification given the predicted and true labels."""
    eq = mb_y == mb_y_hat
    n_correct = eq[mb_y != -1].sum()
    n_total = (mb_y != -1).sum()

    return n_correct * 1.0 / n_total


def main(training_file, training_dir, load_model, skip_train):
    logging.debug('Initializing random seed to 0.')
    random.seed(0)
    np.random.seed(0)

    logging.debug('Loading dataset from: %s' % training_file)
    data = TaggingDataset.load_from_file(training_file)
    logging.debug('Initializing model.')
    tagger = Tagger(data.vocab, data.tags)

    if not skip_train:
        train_data, dev_data = data.split(0.7)

        batches_train = train_data.prepare_batches(n_seqs_per_batch=50)[:-1]
        batches_dev = dev_data.prepare_batches(n_seqs_per_batch=50)[:-1]

        train_mgr = TrainingManager(
            avg_n_losses=len(batches_train),
            training_dir=training_dir,
            tagger_taste_fn=lambda: taste_tagger(tagger, batches_train),
            tagger_dev_eval_fn=lambda: eval_tagger(tagger, batches_dev),
            tagger_save_fn=lambda fname: tagger.save(fname)
        )

        logging.debug('Starting training.')
        while train_mgr.should_continue():
            mb_x, mb_y = random.choice(batches_train)
            mb_loss = tagger.learn(mb_x, mb_y)

            train_mgr.tick(mb_loss=mb_loss)

    evaluate_tagger_and_writeout(tagger)


def evaluate_tagger_and_writeout(tagger):
    stdin = conllu.reader()
    stdout = conllu.writer()
    for sentence in stdin:
        x = []
        for word in sentence:
            x.append(tagger.vocab.get(TaggingDataset.word_obj_to_str(word), tagger.vocab['#OOV']))

        x = np.array([x], dtype='int32')

        y_hat = tagger.predict(x)[0]
        y_hat_str = [tagger.tagset.rev(tag_id) for tag_id in y_hat]

        for word, utag in zip(sentence, y_hat_str):
            word.upos = utag

        stdout.write_sentence(sentence)


if __name__ == '__main__':
    import utils
    utils.init_logging('Tagger')

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('training_file',
                        help='Training file.')
    parser.add_argument('--training_dir',
                        help='Training directory where logs and models will be saved.')
    parser.add_argument('--load_model',
                        help='Model filename.')
    parser.add_argument('--skip_train', action='store_true', default=False,
                        help='Shall we skip training altogether and just use a stored model for tagging?')


    args = parser.parse_args()

    main(**vars(args))
