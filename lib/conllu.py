# This file is part of RH_NNTagging <http://github.com/ufal/rh_nntagging/>.
#
# Copyright 2015 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

class word:
    def __init__(self, form, lemma, upos, lpos, feats, head, deprel, deps, misc):
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.lpos = lpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc

class reader:
    def __init__(self, fname = None):
        import io
        import sys

        if fname is None:
            # Use stdin, but make sure utf-8 encoding is used
            # In Python 2, TextIOWrapper cannot be used as sys.stdin is not BufferedIOBase
            if sys.version_info[0] < 3:
                import codecs
                self.handle = codecs.getreader('utf-8')(sys.stdin)
            else:
                self.handle = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        else:
            self.handle = io.open(fname, mode='r', encoding='utf-8')

    def close(self):
        self.handle.close()

    def next_sentence(self, sentence):
        import re

        # Clear given sentence
        while sentence:
            sentence.pop()

        while True:
            line = self.handle.readline()
            # End on EOF
            if not line: break;
            # Strip newline character
            line = line.rstrip('\n')

            # Skip empty lines and comments before start of sentence
            if not sentence and (not line or line.startswith('#')): continue
            # End sentence after reading an empty line
            if not line: break

            columns = line.split('\t')
            # Skip multi-word tokens
            if columns[0].find('-') >= 0: continue
            # Store the word
            sentence.append(word(
                columns[1] if columns[1] != "_" else "",
                columns[2] if columns[2] != "_" else "",
                columns[3] if columns[3] != "_" else "",
                columns[4] if columns[4] != "_" else "",
                columns[5] if columns[5] != "_" else "",
                int(columns[6]) if columns[6] != "_" else -1,
                columns[7] if columns[7] != "_" else "",
                columns[8] if columns[8] != "_" else "",
                columns[9] if columns[9] != "_" else ""))

        return bool(sentence)

class writer:
    def __init__(self, fname = None):
        import io
        import sys

        if fname is None:
            # Use stdout, but make sure utf-8 encoding is used
            # In Python 2, TextIOWrapper cannot be used as sys.stdout is not BufferedIOBase
            if sys.version_info[0] < 3:
                import codecs
                self.handle = codecs.getwriter('utf-8')(sys.stdout)
            else:
                self.handle = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        else:
            self.handle = io.open(fname, mode='w', encoding='utf-8')

    def close(self):
        self.handle.close()

    def write_sentence(self, sentence):
        index = 1

        for word in sentence:
            self.handle.write('\t'.join([
                str(index),
                word.form or "_",
                word.lemma or "_",
                word.upos or "_",
                word.lpos or "_",
                word.feats or "_",
                str(word.head) if word.head != -1 else "_",
                word.deprel or "_",
                word.deps or "_",
                word.misc or "_"]) + '\n')
            index += 1

        self.handle.write('\n')
