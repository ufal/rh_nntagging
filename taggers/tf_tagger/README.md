# LSTM POS tagger

Bidirectional LSTM tagger with character-based embeddings, implemented with TensorFlow.

The tagger can be run using the following command from the root of the project:

    scripts/run_all.sh -t tf_tagger -n en-dev -dud-1.1/en/en-ud-train.conllu -eud-1.1/en/en-ud-dev.conllu python\ run.py
