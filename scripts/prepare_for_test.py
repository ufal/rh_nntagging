#!/usr/bin/env python

# This file is part of RH_NNTagging <http://github.com/ufal/rh_nntagging/>.
#
# Copyright 2015 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Remove all annotations from CoNLL-U file except for form.
# Reads from files given as arguments and writes to standard output.

import sys
import conllu

stdout = conllu.writer()
sentence = []
for arg in sys.argv[1:]:
    reader = conllu.reader(arg)
    while reader.next_sentence(sentence):
        for word in sentence:
            word.lemma, word.upos, word.lpos, word.feats, word.head, word.deprel, word.deps, word.misc = '', '', '', '', -1, '', '', ''
        stdout.write_sentence(sentence)
