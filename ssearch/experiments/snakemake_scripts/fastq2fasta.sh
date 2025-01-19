#!/usr/bin/env bash

function fq2fa {
    in_fq=$1
    out_fasta=$2

    bname=$(basename $in_fq)

    mkdir -p $(dirname $out_fasta)
    gunzip -c $in_fq |
        paste - - - - |
        cut -f 1,2 |
        sed "s/^@/>${bname}:/" |
        tr "\t" "\n" |
        bgzip -c > $out_fasta
}
export -f fq2fa

set -u

in_fastq=$1
out_fasta=$2

fq2fa $in_fastq $out_fasta
