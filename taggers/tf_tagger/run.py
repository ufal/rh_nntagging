from collections import deque
import random
import os
import Levenshtein
import logging

import numpy as np
import tensorflow as tf
import conllu
from tagger import Tagger
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
        self.max_dev_lemma_perf = 0.0
        self.evals_done = 0
        self.training_dir = training_dir

    def should_continue(self, min_mb_done=None, max_dev_not_improved=None, max_epochs=None):
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
        if max_epochs is None: max_epochs = self.mb_done + 1

        if self.mb_done < min_mb_done:
            return True

        if self.mb_done >= max_epochs * self.n_train_batches:
            return False

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
        logging.debug('Epoch %d, batches %d/%d, avg_loss(%.8f) max_dev_acc(%.5f) curr_dev_acc(%.5f) max_dev_lemma_acc(%.5f) curr_dev_lemma_acc(%.5f)' %
                      (1 + self.mb_done / self.n_train_batches, self.mb_done % self.n_train_batches,
                       self.n_train_batches, np.mean(self.recent_losses), self.max_dev_perf,
                       self.curr_dev_perf, self.max_dev_lemma_perf, self.curr_dev_lemma_perf))

        self.tagger_taste_fn()

    def eval_on_dev(self):
        """Evaluate the model on dev data."""
        self.evals_done += 1
        tagger_perf, lemmatizer_perf = self.tagger_dev_eval_fn()

        if tagger_perf < self.max_dev_perf - self.eps:
            self.not_improved_by_eps_for += 1
        else:
            self.not_improved_by_eps_for = 0

        self.max_dev_perf = max(self.max_dev_perf, tagger_perf)
        self.curr_dev_perf = tagger_perf

        self.max_dev_lemma_perf = max(self.max_dev_lemma_perf, lemmatizer_perf)
        self.curr_dev_lemma_perf = lemmatizer_perf

        if self.training_dir:
            if not os.path.exists(self.training_dir):
                os.makedirs(self.training_dir)

            self.tagger_save_fn(os.path.join(self.training_dir, 'model_%d.pickle' % self.evals_done))

def lemma_from_indices(tagger, lemma_char_indices):
    if tagger.alphabet.rev(lemma_char_indices[0]) == u"<w>":
            lemma_char_indices = lemma_char_indices[1:]
    lemma_chars = []
    for index in lemma_char_indices:
        let = tagger.alphabet.rev(index)
        if let != u"</w>":
            lemma_chars.append(let)
        else:
            break
    return u"".join(lemma_chars)

def taste_tagger(tagger, batches):
    """Print out first 3 examples of the first batch tagged by the model."""
    mb_x, chars, mb_y, lengths, mb_lemma_chars = batches[0]
    mb_y_hat, mb_lemma_chars_hat = \
            tagger.predict_and_eval(mb_x, chars, lengths, mb_y, mb_lemma_chars, out_summaries=False)

    # if there lemmatization is disabled, behave like empty strings have been decoded
    if not mb_lemma_chars_hat:
        mb_lemma_chars_hat = mb_lemma_chars * 0 + tagger.alphabet[u"</w>"]

    for x, y, y_hat, lemma_chars, lemma_chars_hat in \
            zip(mb_x, mb_y, mb_y_hat, mb_lemma_chars, mb_lemma_chars_hat)[:3]:
        line = ' '
        for x_t, y_t, y_hat_t, lemma_chars_t, lemma_chars_hat_t in \
                zip(x, y, y_hat, lemma_chars, lemma_chars_hat)[:5]:
            original_lemma = lemma_from_indices(tagger, lemma_chars_t)
            decoded_lemma = lemma_from_indices(tagger, lemma_chars_hat_t)
            line += ' %s(%s/%s)[%s/%s]' % (tagger.vocab.rev(x_t), tagger.tagset.rev(y_hat_t),
                    tagger.tagset.rev(y_t), original_lemma, decoded_lemma)
        logging.debug(line)


def eval_tagger(tagger, batches):
    """Evaluate the tagger on the given data."""
    acc_total = 0.0
    acc_cnt = 0

    lemma_total = 0.0
    lemma_edit_dist = 0.0
    lemma_len_diff = 0.0
    lemma_count = 0

    for mb_x, chars, mb_y, lengths, mb_lemma_chars in batches:
        mb_y_hat, mb_lemma_chars_hat = tagger.predict_and_eval(mb_x, chars, lengths, mb_y, mb_lemma_chars)

        if mb_lemma_chars_hat:
            for length, lemma_chars, lemma_chars_hat in zip(lengths, mb_lemma_chars, mb_lemma_chars_hat):
                for t in range(min(tagger.num_steps, length)):
                    gt_lemma = lemma_from_indices(tagger, lemma_chars[t])
                    decoded_lemma = lemma_from_indices(tagger, lemma_chars_hat[t])
                    #import sys; print >> sys.stderr, type(gt_lemma)
                    if gt_lemma == decoded_lemma:
                        lemma_total += 1
                    lemma_edit_dist += 1 - Levenshtein.ratio(gt_lemma, decoded_lemma)
                    lemma_len_diff += abs(len(gt_lemma) - len(decoded_lemma))
                    lemma_count += 1

        acc_total += compute_accuracy(mb_y, mb_y_hat)
        acc_cnt += 1

    if hasattr(tagger, 'summary_writer'):
        summary_values = [tf.Summary.Value(tag="tagging_accuracy", simple_value=acc_total / acc_cnt)]
        if lemma_count:
            summary_values += [
                tf.Summary.Value(tag="lemma_accuracy", simple_value=lemma_total / lemma_count),
                tf.Summary.Value(tag="lemma_edit_dist", simple_value=lemma_edit_dist / lemma_count),
                tf.Summary.Value(tag="lemma_length_diff", simple_value=lemma_len_diff / lemma_count)
            ]
        else:
            lemma_count = 1 # to avoid division by zerou in the return statement
        external_str = tf.Summary(value=summary_values)
        tagger.summary_writer.add_summary(external_str, tagger.steps)

    return acc_total / acc_cnt, lemma_total / lemma_count


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
                    num_steps=args.max_sentence_length,
                    optimizer_desc=args.optimizer,
                    generate_lemmas=args.generate_lemmas,
                    l2=args.l2,
                    dropout_prob_values=[float(x) for x in args.dropout.split(",")],
                    experiment_name=args.exp_name)

    batches_train = train_data.prepare_batches(
        args.batch_size, args.max_sentence_length, args.max_word_length)
    # all development data are in one batch
    batches_dev = dev_data.prepare_batches(
        len(dev_data.seqs), args.max_sentence_length, args.max_word_length)

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
        logging.debug("Ctrl+\\ recieved, evaluation will be forced.")
        force_eval["value"] = True
        pass
    signal.signal(signal.SIGQUIT, handle_sigquit)

    logging.debug('Starting training.')
    try:
        permuted_batches = []
        while train_mgr.should_continue(max_epochs=args.max_epochs):
            if not permuted_batches:
                permuted_batches = batches_train[:]
                random.shuffle(permuted_batches)
            words, chars, tags, lengths, lemma_chars = permuted_batches.pop()
            oov_mask = np.vectorize(lambda x: train_data.vocab.count(x) == 1 and np.random.uniform() < args.oov_sampling_p)(words)
            words = np.where(oov_mask, np.zeros(words.shape), words)
            mb_loss = tagger.learn(words, chars, tags, lengths, lemma_chars)

            train_mgr.tick(mb_loss=mb_loss, force_eval=force_eval["value"])
            force_eval["value"] = False
    except KeyboardInterrupt:
        logging.debug("Ctrl+C recieved, stopping training.")

    run_tagger_and_writeout(tagger, dev_data)


def run_tagger_and_writeout(tagger, dev_data):
    logging.debug("Tagging testing data with the trained tagger.")
    for words, chars, _, _ in dev_data.seqs:

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
    parser.add_argument('--optimizer', default='AdamOptimizer(1e-3)', type=str,
                        help='Optimizer specified as class constructor from tf.train module.')
    parser.add_argument('--max-epochs', default=None, type=int,
                        help='Maximum number of training epochs.')
    parser.add_argument('--generate-lemmas', default=False, type=bool,
                        help='Generate lemmas during tagging.')
    parser.add_argument('--l2', default=0.0, type=float,
                        help='L2 regularization.')
    parser.add_argument('--dropout', default="1,1", type=str,
                        help='Dropout keep probability values (formatted as "x,y").')
    parser.add_argument('--exp-name', default="", type=str,
                        help='Experiment name.')

    args = parser.parse_args()

    main(args)
