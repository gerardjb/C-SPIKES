#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --constraint="intel&gpu40"
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --gpu-mps
#SBATCH --no-requeue
# Della selects the partition automatically. Supply --account and measured --time.
set -euo pipefail
manifest=${1:?Supply an absolute frozen manifest path}
output=${2:?Supply an absolute output directory}
shift 2
case "$manifest:$output" in /*:/*) ;; *) echo 'Use absolute manifest and output paths' >&2; exit 2 ;; esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
umask 077
mkdir -p "$output/site"
diagnostics=$(mktemp -d "$output/site/${SLURM_JOB_ID}.XXXXXX")
# Preserve scope/query/log evidence. Slurm owns the service lifecycle.
python -m c_spikes.slurm_mps --output "$diagnostics/scope"
exec python -m c_spikes.pgas_pool --manifest "$manifest" --output "$output" \
  --workers 2 --mps require --cpu-placement separate \
  --mps-service-file "$diagnostics/scope/service-scope.json" "$@"
