#!/usr/bin/env python

import argparse, sys
import tensorflow as tf
import cPickle as pickle

from tagger import tagger_from_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf-model", type=argparse.FileType('r'),
                        help="Serialized TF model.")
    args = parser.parse_args()

    batch_size = 256

    args, vocab, tags, alphabet, params_file = \
        pickle.load(args.tf_model)
    tagger = tagger_from_args(vocab, tags, alphabet, args)

    graph = tf.get_default_graph()

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    print "Variables initialized."

    # TODO load the parameters

    words = []

    def process_batch():
        # TODO process the word batch
        embeddings = sess.run(tagger.char_encoder.outputs, fd=None)
        return embeddings

    for line in sys.stdin:
        words.append(line.rstrip())
        if len(words) == batch_size:
            embeddings = process_batch()
            # TODO print embeddings
            words = []
    process_batch()
