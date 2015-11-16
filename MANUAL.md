# Reading Hachathon - Manual of Neural Network Tagging Project

# Taggers

The individual taggers should be placed in subdirectories of the `taggers`
directory. Taggers can be arbitrary executables, fulfilling the following:
- file with training data is passed as the last argument of the command line
- testing data are read from standard input and the annotated results
  are printed on standard output

The `taggers/simple_dictionary` contains a trivial baseline tagger which uses
for each form the most frequent POS tags from the training data.

# Tagger running scripts

A tagger can be executed (and optionally evaluated) using the
`scripts/run_all.sh` script. This script can execute multiple tagger
configurations on same training and testing data, and optinally evaluate
the reaults. Every configuration can be executed either locally, or on
grid using `qsub`.

The `scripts/run_all.sh` has following usage:
```
  run_all.sh [options] tagger_command_line [tagger_command_line ...]
  Options: -t tagger_name
           -n experiment_name
           -d training_data_file
           -e testing_data_file
           -g
```

The `-t tagger_name` option is mandatory and it specifies a subdirectory of
`taggers` directory containing the tagger to be used.

The `-n experiment_name` option is also mandatory. All outputs of the tagger
are stored in a subdirectory `exp-experiment_name` of the
`taggers/tagger_name` directory.

The `-d training_data_file` option is optional and can be specified multiple
times. It specifies training data relative to the `data` subdirectory of the repo.
All training files are appended together and passed as last argument to the
tagger command.

The `-e testing_data_file` option is optional and can be specified multiple
times. If specifies testing data relative to the `data` subdirectory of the repo.
If used, the specified testing data are cleared (gold annotations removed), fed
to the standard input of the tagger command, and the standard output of the
tagger command is then evaluated against the gold annotations from the testing data.
The evaluation is performed for every testing file separately and written
to standard output.

If `-g` is specified, each tagger command is evaluated on the grid using `qsub`.

All remaining arguments are interpreted as tagger command lines -- each argument
is one command to execute. For every command line, files containing standard
output and standard error are created in the experiment subdirectory (using names
derived from the tagger command line). Every tagger command is executed in the
tagger directory (i.e., `taggers/tagger_name` directory of the repo) and the
`conllu.py` module (described later) is in Python library path.

*Example:* To traing `simple_dictonary.py` tagger on English, evaluate results on development data and store the results in `exp-en-dev` directory, run
```
  scripts/run_all.sh -t simple_dictionary -n en-dev -dud-1.1/en/en-ud-train.conllu -eud-1.1/en/en-ud-dev.conllu simple_dictionary.py
```

*Example:* To train `simple_dictonary.py` tagger on all Czech training data (there are four), evaluate results on both development and testing data and store the results in `exp-cs` directory, using our cluster, run
```
  scripts/run_all.sh -g -t simple_dictionary -n cs -dud-1.1/cs/cs-ud-train-{c,l,m,v}.conllu -eud-1.1/cs/cs-ud-{dev,test}.conllu simple_dictionary.py
```

# Python conllu.py module

Python module `conllu.py` provides methods for loading and saving CoNLL-U files.
The module is located in `lib` subdirectory, which is added to `PYTHONPATH`
automatically when running a tagger using the `scripts/run_all.sh` script.

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
