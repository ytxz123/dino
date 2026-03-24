#!/usr/bin/env bash

set -euo pipefail

# 一键运行默认全流程。
# 如果不传环境变量，就走 build-all 的内置默认参数。
# 如果需要改原始数据或输出路径，优先通过环境变量覆盖。

DATASET_ROOT="${DATASET_ROOT:-/dataset/zsy/dataset-extracted}"
TRAIN_ROOT="${TRAIN_ROOT:-}"
VAL_ROOT="${VAL_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-dataset_generation_output/processed_all}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
LANE_ONLY="${LANE_ONLY:-0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=( "${PYTHON_BIN}" )
elif [[ -x "/opt/anaconda3/envs/data/bin/python" ]]; then
  PYTHON_CMD=( "/opt/anaconda3/envs/data/bin/python" )
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=( conda run -n data python )
else
  PYTHON_CMD=( python )
fi

ARGS=( -m dataset_generation.cli build-all )

if [[ -n "${OUTPUT_ROOT}" ]]; then
  ARGS+=( --output-root "${OUTPUT_ROOT}" )
fi

if [[ -n "${MANIFEST_PATH}" ]]; then
  ARGS+=( --manifest-path "${MANIFEST_PATH}" )
fi

if [[ -n "${TRAIN_ROOT}" ]]; then
  ARGS+=( --train-root "${TRAIN_ROOT}" )
fi

if [[ -n "${VAL_ROOT}" ]]; then
  ARGS+=( --val-root "${VAL_ROOT}" )
fi

if [[ -z "${TRAIN_ROOT}" && -z "${VAL_ROOT}" ]]; then
  ARGS+=( --dataset-root "${DATASET_ROOT}" )
fi

if [[ "${LANE_ONLY}" == "1" || "${LANE_ONLY}" == "true" || "${LANE_ONLY}" == "TRUE" ]]; then
  ARGS+=( --lane-only )
fi

if [[ "$#" -gt 0 ]]; then
  ARGS+=( "$@" )
fi

exec "${PYTHON_CMD[@]}" "${ARGS[@]}"