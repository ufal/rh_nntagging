#!/bin/bash

set -e

[ $# -lt 3 ] && { echo "Usage: $0 iterations order training_data <testing_data" >&2; exit 1; }
iterations="$1"
order="$2"
data="$3"

cat | awk '
  /^#/{next}
  /^[0-9]*-/{next}
  /^$/{print;next}

  BEGIN{tags_len = split("ADJ ADP ADV AUX CONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X", tags, " ")}
  {a=$2; for (i = 1; i <= tags_len; i++) a = a " " $2 " " tags[i]; print a}
' | morphodita/run_tagger --input=vertical --output=vertical <(cat "$data" | awk '
  /^#/{next}
  /^[0-9]*-/{next}
  /^$/{print;next}

  BEGIN{tags_len = split("ADJ ADP ADV AUX CONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X", tags, " ")}
  {a=$2; for (i = 1; i <= tags_len; i++) a = a " " $2 " " tags[i]; print a "\t" $2 "\t" $4}
' | morphodita/train_tagger generic$order morphodita/external.dict 0 morphodita/feature_sequences-$order.txt "$iterations" 0) | awk '
  BEGIN{id=1;FS="\t"}

  /^$/{id=1;print;next}
  {print id "\t" $2 "\t" $2 "\t" $3 "\t_\t_\t_\t_\t_\t_"; id += 1}
'
