#!/usr/bin/env bash
#SBATCH --job-name=forB4Lecture    # Job name
#SBATCH --mem=32G                   # Memory total in GB (for all cores)
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --output=batchlog/%j/job_output_%j.log  # output.log
#SBATCH --error=batchlog/%j/job_error_%j.log    # error.log
#SBATCH --gres=gpu:1                # Number of GPUs to use
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON=${PYTHON:-python}

# --- 設定 ---
DOWNLOAD_DIR=${DOWNLOAD_DIR:-feature_cache/cmu_arctic_full}
MANIFEST=${MANIFEST:-feature_cache/cmu_arctic_full/feature_manifest.jsonl}
FEATURE_OUTPUT_DIR=${FEATURE_OUTPUT_DIR:-feature_cache/cmu_arctic_full}
SPEAKERS=${SPEAKERS:-bdl,slt}
CONDITIONS=${CONDITIONS:-hubert_soft,world_aux}
SPLIT=${SPLIT:-all}

# GPU/CPU の選択
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  GPU_ARGS=(--gpu 0)
else
  GPU_ARGS=(--gpu 0)
fi

# --- Step 1: マニフェスト再生成（ダウンロードはスキップ）---
echo "=== Step 1: prepare_speech_data (再生成) ==="
"${PYTHON}" scripts/prepare_speech_data.py \
  --root "${DOWNLOAD_DIR}" \
  --manifest "${MANIFEST}" \
  --speakers "${SPEAKERS}"

# --- Step 1.5: CSV→JSONL変換 ---
echo "=== Step 1.5: CSV -> JSONL 変換 ==="
"${PYTHON}" - <<EOF
import csv, json, io
from pathlib import Path

manifest = Path("${MANIFEST}")
text = manifest.read_text(encoding="utf-8")

first_line = text.splitlines()[0].strip()
try:
    json.loads(first_line)
    print("すでにJSONL形式です。変換をスキップします。")
except json.JSONDecodeError:
    reader = csv.DictReader(io.StringIO(text))
    jsonl_lines = [json.dumps(row, ensure_ascii=False) for row in reader]
    manifest.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
    print(f"変換完了: {len(jsonl_lines)} 行")
EOF

# --- Step 2: 特徴量抽出（split=all で全データ）---
MAX_UTTERANCES=${MAX_UTTERANCES:-2000}
echo "=== Step 2: extract_speech_features (split=${SPLIT}, max-utterances=${MAX_UTTERANCES}) ==="
"${PYTHON}" scripts/extract_speech_features.py \
  --manifest "${MANIFEST}" \
  --output-dir "${FEATURE_OUTPUT_DIR}" \
  --conditions "${CONDITIONS}" \
  --split "${SPLIT}" \
  --speakers "${SPEAKERS}" \
  --max-utterances "${MAX_UTTERANCES}" \
  "${GPU_ARGS[@]}"

echo "=== Done! ==="
echo "マニフェスト: ${MANIFEST}"
echo "特徴量出力先: ${FEATURE_OUTPUT_DIR}"