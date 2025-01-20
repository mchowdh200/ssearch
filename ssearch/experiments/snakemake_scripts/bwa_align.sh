#!/usr/bin/env bash

function get_ext {
    local f=$1
    x="${f##*.}"
    if [ "$x" = "gz" ]; then
        y="${f%.*}"
        y="${y##*.}"
        x="$y.$x"
    fi
    echo $x
}

while getopts "q:r:o:t:" opt; do
    case $opt in
        q) query_fq=$OPTARG ;;
        r) ref_fasta=$OPTARG ;;
        o) output_bam=$OPTARG ;;
        t) threads=$OPTARG ;;
    esac
done

bwa mem -t $threads $ref_fasta $query_fq |
    samtools view --no-header -b - > $output_bam


