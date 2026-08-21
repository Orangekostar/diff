#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
registered_config="${project_root}/paper_v3/configs/d8_residual_diffusion.yaml"
test -f "${registered_config}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${project_root}/src"
export CUDA_VISIBLE_DEVICES=0,1,2
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

exec /home/ww/miniconda3/bin/python \
  "${project_root}/scripts/run_d8_residual_diffusion.py" "$@"
