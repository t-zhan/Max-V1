#!/usr/bin/env bash

DATA_DIR="./data"
WORKERS=8
VQA_DATASET="Telkwevr/Bench2Drive-VL-base"
RAW_DATASET="rethinklab/Bench2Drive"
VQA_DIRNAME="Bench2Drive-VL-base"
RAW_DIRNAME="Bench2Drive"
RAW_ARCHIVE_DIR="$DATA_DIR/${RAW_DIRNAME}-archives"
VQA_DIR="$DATA_DIR/$VQA_DIRNAME"
RAW_DIR="$DATA_DIR/$RAW_DIRNAME"

mkdir -p "$RAW_ARCHIVE_DIR" "$VQA_DIR" "$RAW_DIR"

echo "Downloading raw data..."
hf download "$RAW_DATASET" \
  --repo-type dataset \
  --local-dir "$RAW_ARCHIVE_DIR" \
  --max-workers 1

echo "Downloading VQA data..."
hf download "$VQA_DATASET" \
  --repo-type dataset \
  --local-dir "$VQA_DIR" \
  --max-workers 1

echo "Unpacking raw data with $WORKERS workers..."
find "$RAW_ARCHIVE_DIR" -name "*.tar.gz" -print0 \
  | xargs -0 -r -n 1 -P "$WORKERS" bash -c '
  archive="$1"
  raw_dir="$2"
  tar -xzf "$archive" -C "$raw_dir"
  rm "$archive"
' _ {} "$RAW_DIR"

rmdir "$RAW_ARCHIVE_DIR"

echo "Data is ready at $DATA_DIR"
