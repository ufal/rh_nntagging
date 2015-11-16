# LSTM POS tagger

Feedforward LSTM tagger implemented with Chainer.

The tagger can be run using the following command from the root of the project:

    scripts/run_all.sh -t lstm_tagger -n en-dev -dud-1.1/en/en-ud-train.conllu -eud-1.1/en/en-ud-dev.conllu "python run.py"
