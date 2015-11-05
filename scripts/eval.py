#!/usr/bin/env python

# This file is part of RH_NNTagging <http://github.com/ufal/rh_nntagging/>.
#
# Copyright 2015 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Evaluates tagging accuracy on given CoNLL-U files.
# The gold files are given on the command line,
# system-generated files are on standard input.

import os
import sys
import conllu

stdin = conllu.reader()
sentence_gold, sentence_system = [], []

for arg in sys.argv[1:]:
    # Evaluate for each file separately
    total, lemma, upos, uposfeats, lpos, every = 0, 0, 0, 0, 0, 0

    reader = conllu.reader(arg)
    while reader.next_sentence(sentence_gold):
        stdin.next_sentence(sentence_system)
        for i in range(max(len(sentence_gold), len(sentence_system))):
            gold_form = sentence_gold[i].form if i < len(sentence_gold) else None
            system_form = sentence_system[i].form if i < len(sentence_system) else None
            if gold_form != system_form:
                raise Exception("Forms '{}' from gold file {} and '{}' from corresponding system file do not match".format(str(gold_form), arg, str(system_form)))

            total += 1
            lemma += sentence_gold[i].lemma == sentence_system[i].lemma
            upos += sentence_gold[i].upos == sentence_system[i].upos
            uposfeats += sentence_gold[i].upos == sentence_system[i].upos and sentence_gold[i].feats == sentence_system[i].feats
            lpos += sentence_gold[i].lpos == sentence_system[i].lpos
            every += sentence_gold[i].lemma == sentence_system[i].lemma and sentence_gold[i].upos == sentence_system[i].upos and sentence_gold[i].feats == sentence_system[i].feats and sentence_gold[i].lpos == sentence_system[i].lpos

    print "File {}".format(os.path.basename(arg))
    print "  Total: {:6.2f}".format(100. * every / total)
    print "  Lemma: {:6.2f}".format(100. * lemma / total)
    print "  UPos:  {:6.2f}".format(100. * upos / total)
    print "  UPosF: {:6.2f}".format(100. * uposfeats / total)
    print "  LPos:  {:6.2f}".format(100. * lpos / total)
