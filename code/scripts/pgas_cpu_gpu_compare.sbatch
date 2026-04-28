#!/bin/bash
#SBATCH --job-name=pgas_cpu_gpu_compare
#SBATCH --output=logs/pgas_cpu_gpu_compare_%j.out
#SBATCH --error=logs/pgas_cpu_gpu_compare_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=gerardjb@princeton.edu

set -euo pipefail

# Load modules / activate env
set +u
module load anaconda3/2024.6
export PS1=${PS1-}
source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh" || true
conda activate c_spikes_e
set -u

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-threads}

DATASET=${DATASET:-data/janelia_8f/excitatory/jGCaMP8f_ANM471994_cell04.mat}
RUN_NAME=${RUN_NAME:-jGCaMP8f_cpu_gpu_compare_$(date +%Y%m%d_%H%M%S)}

python scripts/compare_pgas_cpu_gpu.py \
  --dataset "$DATASET" \
  --trial-index 0 \
  --snippet-duration-s 10 \
  --niter 200 \
  --run-name "$RUN_NAME"
