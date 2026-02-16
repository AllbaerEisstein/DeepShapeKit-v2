#!/usr/bin/env bash
set -e

usage() {
  echo "Usage: $0 <input.mp4> <start_frame> <end_frame> [-fr <fps>] [-cropx <px>] [-cropy <px>]"
  exit 1
}

INPUT="$1"
START_FRAME="$2"
END_FRAME="$3"
shift 3

# Optional args
OUT_FPS=""
CROP_X=0
CROP_Y=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -fr) OUT_FPS="$2"; shift 2 ;;
    -cropx) CROP_X="$2"; shift 2 ;;
    -cropy) CROP_Y="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

if [ -z "$INPUT" ] || [ -z "$START_FRAME" ] || [ -z "$END_FRAME" ]; then
  usage
fi

# Frame rate of source
FPS=$(ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate "$INPUT")
FPS=$(awk -F/ '{printf "%.6f", $1/$2}' <<< "$FPS")

# Time values
START_TIME=$(awk -v sf="$START_FRAME" -v fps="$FPS" 'BEGIN { printf "%.6f", sf/fps }')
DURATION=$(awk -v sf="$START_FRAME" -v ef="$END_FRAME" -v fps="$FPS" 'BEGIN { printf "%.6f", (ef - sf + 1)/fps }')

# Crop filter
CROP_FILTER=""
if [ "$CROP_X" -gt 0 ] || [ "$CROP_Y" -gt 0 ]; then
  # Compute center crop: out_w = in_w - cropx, out_h = in_h - cropy
  # x offset = cropx/2, y offset = cropy/2
  CROP_FILTER="crop=in_w-${CROP_X}:in_h-${CROP_Y}:${CROP_X}/2:${CROP_Y}/2"
fi

# Frame rate filter
FPS_FILTER=""
if [ -n "$OUT_FPS" ]; then
  if [ -n "$CROP_FILTER" ]; then
    FPS_FILTER=",fps=${OUT_FPS}"
  else
    FPS_FILTER="fps=${OUT_FPS}"
  fi
fi

# Combine filters
FILTERS=""
if [ -n "$CROP_FILTER" ] || [ -n "$FPS_FILTER" ]; then
  FILTERS="-vf \"${CROP_FILTER}${FPS_FILTER}\""
fi

# Output
B="$(basename "$INPUT")"
NAME="${B%.*}"
OUT="${NAME}_${START_FRAME}-${END_FRAME}.mp4"

# Build FFmpeg command
# shellcheck disable=SC2086
CMD="ffmpeg -ss $START_TIME -i \"$INPUT\" -t $DURATION \
  $FILTERS -c:v libx264 -pix_fmt yuv420p -c:a aac \"$OUT\""

echo "Running:"
echo "$CMD"
eval "$CMD"

echo "Saved: $OUT"
