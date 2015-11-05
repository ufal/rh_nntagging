# Reading Hachathon - Manual of Neural Network Tagging Project

# Python conllu.py module

Python module `conllu.py` provides methods for loading and saving CoNLL-U files.
The module is located in `lib` subdirectory, which is added to `PYTHONPATH`
automatically when running a tagger using the `scripts/run_tagger.sh` script.

The `conllu.py` module provides the following classes:

- `word`

  The `word` class represents a CoNLL-U word. It has the following data fields:
  - `form`
  - `lemma`
  - `upos`: universal part-of-speech
  - `lpos`: language-specific part-of-speech
  - `feats`: morphological features
  - `head`: integer
  - `deprel`
  - `deps`
  - `misc`
  Fields with no value are empty (not underscores as in CoNLL-U), except for
  the `head` fields, which is negative when it has no value.

  Note that to avoid encoding problems, all string are expected be represented
  using the `unicode` type on Python 2.

- `reader`

  The `reader` class allows reading sentences from a CoNLL-U file. It has the
  following methods:
  - `__init__(self, fname = None)`: Create a `reader` instance, reading
    from the specified file. If no filename is given, standard input is used
    insted. Note that UTF-8 encoding is always used, even if default encoding
    of standard input might be different.
  - `next_sentence(self, sentence)`: Loads one sentence from the file. The
    `sentence` must be a list, which is filled with a sequence of `word`
    instances. The return value is `False` when end-of-file is reached, and
    `True` otherwise.
  - `close`: Close the file.

- `writer`

  The `writer` class allows writing sentences to a CoNLL-U file. It has the
  following methods:
  - `__init__(self, fname = None)`: Create a `writer` instance, writing
    to the specified file. If no filename is given, standard output is used
    insted. Note that UTF-8 encoding is always used, even if default encoding
    of standard output might be different.
  - `writer_sentence(self, sentence)`: Write the sentence to the file.
    The `sentence` must be a list of `word` instances.
  - `close`: Close the file.
