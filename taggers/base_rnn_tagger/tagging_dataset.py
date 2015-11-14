from collections import Counter
import math
import numpy as np
import random

import conllu
from vocab import Vocab


class TaggingDataset(object):
    def __init__(self, seqs, vocab, label_vocab):
        self.seqs = seqs
        self.vocab = vocab
        self.tags = label_vocab

    def get_n_tags(self):
        return len(self.tags)

    def shuffle(self):
        random.shuffle(self.seqs)

    def split(self, ratio):
        pivot = int(ratio * len(self.seqs))
        seqs1 = self.seqs[:pivot]
        seqs2 = self.seqs[pivot:]
        d1 = TaggingDataset(seqs1, self.vocab, self.tags)
        d2 = TaggingDataset(seqs2, self.vocab, self.tags)

        return d1, d2

    def prepare_batches(self, n_seqs_per_batch=-1):
        res = []

        if n_seqs_per_batch == -1:
            n_seqs_per_batch = len(self.seqs)

        n_batches = int(math.ceil(len(self.seqs) * 1.0 / n_seqs_per_batch))
        for seq_id in range(n_batches):
            batch = self.seqs[seq_id * n_seqs_per_batch:(seq_id + 1) * n_seqs_per_batch]
            res.append(self._batch_to_numpy(batch))

        return res

    def _batch_to_numpy(self, batch):
        max_seq_len = None
        for x, y in batch:
            max_seq_len = max(len(x), max_seq_len)

        res_x = np.zeros((max_seq_len, len(batch)), dtype='int32')
        res_y = np.zeros((max_seq_len, len(batch)), dtype='int32') - 1  # Chainer's softmax loss understands -1 as ignore this label.

        for i, (x, y) in enumerate(batch):
            res_x[:len(x), i] = x
            res_y[:len(y), i] = y

        return res_x.T, res_y.T

    @staticmethod
    def load_from_file(fname, vocab=None, tags=None):
        reader = conllu.reader(fname)

        learn_tags, learn_vocab, tags, vocab = TaggingDataset.initialize_vocab_and_tags(tags, vocab)

        seqs = []
        for sentence in reader:
            x = []
            y = []
            for word in sentence:
                word_id, tag_id = TaggingDataset.get_word_and_tag_id(word, vocab, tags,
                                                                     learn_vocab, learn_tags)

                x.append(word_id)
                y.append(tag_id)

            seqs.append((x, y))

        res = TaggingDataset(seqs, vocab, tags)

        return res

    @staticmethod
    def word_obj_to_str(word):
        return word.form

    @staticmethod
    def initialize_vocab_and_tags(tags, vocab):
        if not vocab:
            vocab = Vocab()
            vocab.add('#OOV')
            learn_vocab = True
        else:
            learn_vocab = False
        if not tags:
            tags = Vocab()
            learn_tags = True
        else:
            learn_tags = False
        return learn_tags, learn_vocab, tags, vocab

    @staticmethod
    def get_word_and_tag_id(word, vocab, tags, learn_vocab, learn_tags):
        if learn_vocab:
            word_id = vocab.add(TaggingDataset.word_obj_to_str(word))
        else:
            word_id = vocab.get(TaggingDataset.word_obj_to_str(word), vocab['#OOV'])
        if learn_tags:
            tag_id = tags.add(word.upos)
        else:
            tag_id = tags[word.upos]
        return word_id, tag_id


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