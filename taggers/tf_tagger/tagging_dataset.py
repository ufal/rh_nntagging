from collections import Counter
import math
import numpy as np
import random

import conllu
from vocab import Vocab


class TaggingDataset(object):
    """Holds a dataset for tagging."""
    def __init__(self, seqs, vocab, alphabet, label_vocab):
        """Initialize new dataset.

        Args:
            seqs: list of (sentence, tags) pairs that contain ids of words and tags
            vocab: Vocab mapping between words and ids
            label_vocab: Vocab mapping between tags and ids
        """
        self.seqs = seqs
        self.vocab = vocab
        self.alphabet = alphabet
        self.tags = label_vocab

    def get_n_tags(self):
        """How many tags are there?"""
        return len(self.tags)

    def shuffle(self):
        """Shuffle the order of the sequences."""
        random.shuffle(self.seqs)

    def split(self, ratio):
        """Split the dataset into two datasets given the ratio.

        The first one contains `ratio` percent and the other the remainder.
        WARNING: The vocabularies are not rebuilt and remain the same.
        """

        pivot = int(ratio * len(self.seqs))
        seqs1 = self.seqs[:pivot]
        seqs2 = self.seqs[pivot:]
        d1 = TaggingDataset(seqs1, self.vocab, self.alphabet, self.tags)
        d2 = TaggingDataset(seqs2, self.vocab, self.alphabet, self.tags)

        return d1, d2

    def prepare_batches(self, batch_size, max_seq_len, max_word_len):
        """Return a list of minibatches created out of the dataset.

        Args:
            batch_size: how many sequences should each minibatch contain?
        """
        res = []

        n_batches = int(math.ceil(len(self.seqs) / float(batch_size)))
        for seq_id in range(n_batches):
            batch = self.seqs[seq_id * batch_size:(seq_id + 1) * batch_size]
            res.append(self._batch_to_numpy(batch, max_seq_len, max_word_len, batch_size))

        return res

    def _batch_to_numpy(self, batch, max_seq_len, max_word_len, batch_size):
        """
        Convert given minibatch to numpy arrays with appropriate padding
        After this, the vocabulary indices are shifted by one where the
        zero index means the padding.
        """

#        for x, y in batch:
#            max_seq_len = max(len(x), max_seq_len)

        res_x = np.zeros((batch_size, max_seq_len), dtype='int32')
        res_c = np.zeros((batch_size, max_seq_len, max_word_len), dtype='int32')
        res_y = np.zeros((batch_size, max_seq_len), dtype='int32')
        # this + 2 in lemma characters is for the special start and end symbols
        res_lemma_c = np.zeros((batch_size, max_seq_len, max_word_len + 2), dtype='int32')
        lengths = np.zeros((batch_size), dtype='int64')

        for i, (words, chars, tags, lemma_chars) in enumerate(batch):
            res_x[i, :min(len(words), max_seq_len)] = words[:min(len(words), max_seq_len)]
            res_y[i, :min(len(tags), max_seq_len)] = tags[:min(len(tags), max_seq_len)]

            for j in range(min(len(words), max_seq_len)):
                res_c[i, j, :min(len(chars[j]), max_word_len)] = chars[j][:min(len(chars[j]), max_word_len)]
                res_lemma_c[i, j, :min(len(chars[j]), max_word_len + 2)] = \
                        chars[j][:min(len(chars[j]), max_word_len + 2 )]


            lengths[i] = len(words)

        return res_x, res_c, res_y, lengths, res_lemma_c

    @staticmethod
    def load_from_file(fname, vocab=None, alphabet=None, tags=None):
        """Load dataset from the given file."""
        reader = conllu.reader(fname)

        learn_tags, learn_vocab, tagset, vocab, alphabet = TaggingDataset.initialize_vocab_and_tags(tags, vocab, alphabet)

        seqs = []
        for sentence in reader:
            words = []
            tags = []
            chars = []
            lemma_chars = []

            for word in sentence:
                word_id, char_ids, tag_id, lemma_char_ids = \
                        TaggingDataset.get_word_and_tag_id(word, vocab, alphabet, tagset,
                                                           learn_vocab, learn_tags)

                words.append(word_id)
                chars.append(char_ids)
                tags.append(tag_id)
                lemma_chars.append(lemma_char_ids)

            seqs.append((words, chars, tags, lemma_chars))

        res = TaggingDataset(seqs, vocab, alphabet, tagset)

        return res

    @staticmethod
    def word_obj_to_str(word):
        """Stringify the word for purposes of tagging."""
        return word.form

    @staticmethod
    def initialize_vocab_and_tags(tags, vocab, alphabet):
        if not vocab:
            vocab = Vocab()
            vocab.add('#OOV')
            alphabet = Vocab()
            alphabet.add('#OOA')

            learn_vocab = True
        else:
            learn_vocab = False
        if not tags:
            tags = Vocab()
            tags.add("#OOT")
            learn_tags = True
        else:
            learn_tags = False


        return learn_tags, learn_vocab, tags, vocab, alphabet

    @staticmethod
    def get_word_and_tag_id(word, vocab, alphabet, tags, learn_vocab, learn_tags):
        word_text = TaggingDataset.word_obj_to_str(word)
        chars = list(word_text)
        lemma_chars = ["<w>"]+list(word.lemma)+["</w>"]

        if learn_vocab:
            word_id = vocab.add(word_text)
            char_ids = [alphabet.add(char) for char in chars]
            lemma_char_ids = [alphabet.add(char) for char in lemma_chars]
        else:
            word_id = vocab.get(word_text, vocab['#OOV'])
            char_ids = [alphabet.get(char, alphabet['#OOA']) for char in chars]
            lemma_char_ids = [alphabet.get(char, alphabet['#OOA']) for char in lemma_chars]
        if learn_tags:
            tag_id = tags.add(word.upos)
        else:
            tag_id = tags[word.upos]
        return word_id, char_ids, tag_id, lemma_char_ids


def main(fname, split, dont_shuffle, fout1, fout2):
    ds = TaggingDataset.load_from_file(fname)
    if not dont_shuffle:
        ds.shuffle()

    ratio = float(split)

    ds1, ds2 = ds.split(ratio)

    ds1.save_to_file(fout1)
    ds2.save_to_file(fout2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--dont_shuffle', type=bool, default=False)
    parser.add_argument('--fout1')
    parser.add_argument('--fout2')

    args = parser.parse_args()

    main(**vars(args))
