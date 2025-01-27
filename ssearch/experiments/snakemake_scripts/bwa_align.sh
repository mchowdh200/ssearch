#!/usr/bin/env bash

# function get_ext {
#     local f=$1
#     x="${f##*.}"
#     if [ "$x" = "gz" ]; then
#         y="${f%.*}"
#         y="${y##*.}"
#         x="$y.$x"
#     fi
#     echo $x
# }

while getopts "q:r:o:t:" opt; do
    case $opt in
        l) fq_1=$OPTARG ;;
        r) fq_2=$OPTARG ;;
        f) ref_fasta=$OPTARG ;;
        o) output_bam=$OPTARG ;;
        t) threads=$OPTARG ;;
    esac
done

bwa mem -t $threads $ref_fasta $fq_1 $fq_2 |
    samtools view -b - > $output_bam


