#!/bin/bash

# This file is part of RH_NNTagging <http://github.com/ufal/rh_nntagging/>.
#
# Copyright 2015 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

dir=`dirname "$0"`

usage() {
  [ -n "$1" ] && echo "$1!" >&2
  echo "Usage: $0 [options] tagger command line" >&2
  echo "Options: -t tagger_name (directory name in taggers directory)" >&2
  echo "         -n experiment_name (directory created in specified tagger)" >&2
  echo "         -d training_data_file (relative to data directory)" >&2
  echo "         -e testing_data_file (relative to data directory)" >&2
  exit 1
}

tagger=""
experiment=""
training=()
testing=()
grid=""

while getopts "n:t:d:e:g" opt; do
  case $opt in
    t) [ -z "$tagger" ] || usage "The tagger name was specified multiple times"
       tagger="$OPTARG";;
    n) [ -z "$experiment" ] || usage "The experiment name was specified multiple times"
       experiment="$OPTARG";;
    d) training+=("../../data/$OPTARG");;
    e) testing+=("../../data/$OPTARG");;
    *) usage
  esac
done
shift $((OPTIND-1))
[ $# -eq 0 ] && usage "No tagger command was given"
[ -z "$tagger" ] && usage "No tagger name was given"
[ -z "$experiment" ] && usage "No experiment name was given"

[ -d "$dir/../taggers/$tagger" ] || { echo "The given tagger does not exist in taggers directory!" >&2; exit 1; }

cd "$dir/../taggers/$tagger"
mkdir -p exp-"$experiment"

description="$*"
description="${description// /}"
description="${description//\//}"
if [ -z "${testing[0]}" ]; then
  PATH=.:"$PATH" PYTHONPATH=../../lib:"$PYTHONPATH" "$@" "${training[@]}" >exp-"$experiment/$description".out 2>exp-"$experiment/$description".err
else
  (PYTHONPATH=../../lib:"$PYTHONPATH" ../../scripts/prepare_for_test.py "${testing[@]}" | PATH=.:"$PATH" PYTHONPATH=../../lib:"$PYTHONPATH" "$@" "${training[@]}" | PYTHONPATH=../../lib:"$PYTHONPATH" ../../scripts/eval.py "${testing[@]}") >exp-"$experiment/$description".out 2>exp-"$experiment/$description".err
fi
