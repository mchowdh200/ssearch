#!/usr/bin/env bash

set -eu

f1=$1
f2=$2
o=$3

cat $f1 $f2 > $o
samtools faidx $o
