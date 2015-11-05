#!/usr/bin/env python

# This file is part of RH_NNTagging <http://github.com/ufal/rh_nntagging/>.
#
# Copyright 2015 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import collections
import sys

import conllu

dictionary = {}

# Process all arguments as training files
for arg in sys.argv[1:]:
    reader = conllu.reader(arg)
    sentence = []
    while reader.next_sentence(sentence):
        for word in sentence:
            dictionary.setdefault(word.form, collections.defaultdict(lambda:0))["\t".join([word.lemma, word.upos, word.lpos, word.feats])] += 1

# Find most frequent analysis, using the lexicographically smaller when equal
for form in dictionary:
    best, best_count = '', 0
    for analysis, count in dictionary[form].iteritems():
        if count > best_count or (count == best_count and analysis < best):
            best, best_count = analysis, count
    dictionary[form] = best

# Analyse all data passed on standard input to standard output
stdin = conllu.reader()
stdout = conllu.writer()
sentence = []
while stdin.next_sentence(sentence):
    for word in sentence:
        word.lemma, word.upos, word.lpos, word.feats = dictionary.get(word.form, '\t\t\t').split('\t')
    stdout.write_sentence(sentence)
