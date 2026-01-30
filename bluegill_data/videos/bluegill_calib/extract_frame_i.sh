#!/usr/bin/env bash

# default frame
FRAME=0

# parse -f option
while getopts "f:" opt; do
    case $opt in
        f) FRAME=$OPTARG ;;
        *) echo "Usage: $0 [-f frame_number] videos..." ; exit 1 ;;
    esac
done

# shift parsed options so "$@" are just the video files
shift $((OPTIND - 1))

for v in "$@"; do
    ffmpeg -y -i "$v" -vf "select=eq(n\,${FRAME})" -frames:v 1 "${v%.*}_frame${FRAME}.png"
done

