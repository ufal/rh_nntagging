#!/usr/bin/env perl
# This file is part of RH_NNTagging <http://github.com/ufal/rh_nntagging/>.
#
# Copyright 2015 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

use warnings;
use strict;
use utf8;
use open qw(:std :utf8);

@ARGV >= 1 or die "Usage: $0 evaluation_type(Total|Lemma|UPos|UPosF|LPos)\n";
my $evaluation_type = shift @ARGV;

my @results = ();

# Collect results
foreach my $file (glob("*.out")) {
  my $testing_file = '';

  open (my $f, "<", $file) or die "Cannot open file $file: $!";
  while (<$f>) {
    chomp;
    /^File (.*)$/ and $testing_file = $1;
    /^\s*$evaluation_type:\s*([0-9.]*)$/ and push @results, {score=>$1, file=>$testing_file, opts=>$file};
  }
  close $f;
}

@results or exit;

# Remove common prefix of opts
my $prefix = $results[0]->{opts};
foreach my $res (@results) {
  my ($opts, $common_length) = ($res->{opts}, 0);
  $common_length++ while ($common_length < length($prefix) && $common_length < length($opts) && substr($prefix, $common_length, 1) eq substr($opts, $common_length, 1));
  $prefix = substr($prefix, 0, $common_length);
}

foreach my $res (@results) {
  $res->{opts} = substr($res->{opts}, length($prefix));
}

# Write sorted results
foreach my $res (@results = sort {$b->{score} <=> $a->{score}} @results) {
  print "$res->{score} $res->{file} $res->{opts}\n";
}
