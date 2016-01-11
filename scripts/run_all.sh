#!/bin/bash

# This file is part of RH_NNTagging <http://github.com/ufal/rh_nntagging/>.
#
# Copyright 2015 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

trap '' SIGTSTP # Trap Ctrl+Z so it can be catched by the tagger itself.

dir=`dirname "$0"`

usage() {
  [ -n "$1" ] && echo "$1!" >&2
  echo "Usage: $0 [options] tagger_command_line [tagger_command_line ...]" >&2
  echo "Options: -t tagger_name (directory name in taggers directory)" >&2
  echo "         -n experiment_name (directory created in specified tagger)" >&2
  echo "         -d training_data_file (relative to data directory)" >&2
  echo "         -e testing_data_file (relative to data directory)" >&2
  echo "         -g (run on grid using qsub)" >&2
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
    d) training+=("-d$OPTARG");;
    e) testing+=("-e$OPTARG");;
    g) grid="1";;
    *) usage
  esac
done
shift $((OPTIND-1))
[ $# -eq 0 ] && usage "No tagger command line was given"
[ -z "$tagger" ] && usage "No tagger name was given"
[ -z "$experiment" ] && usage "No experiment name was given"

[ -d "$dir/../taggers/$tagger" ] || { echo "The given tagger does not exist in taggers directory!" >&2; exit 1; }
mkdir -p "$dir/../taggers/$tagger/exp-$experiment"

for command_line in "$@"; do
  description="${command_line// /}"
  description="${description//\//}"
  description="${description//,/}"
  log="$dir/../taggers/$tagger/exp-$experiment/$description"

  if [ -z "$grid" ]; then
    "$dir"/run_tagger.sh -t"$tagger" -n"$experiment" "${training[@]}" "${testing[@]}" $command_line 2> >(trap "" SIGINT; tee "$log.err" >&2) | (trap "" SIGINT; tee "$log".out)
  else
    >"$log".out
    >"$log".err
    qsub $SGE_ARGS -N "${tagger}_exp-${experiment}" -o "$log".out -e "$log".err -b y -cwd "$dir"/run_tagger.sh -t"$tagger" -n"$experiment" "${training[@]}" "${testing[@]}" $command_line
  fi
done
