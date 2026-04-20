#!/bin/bash
# Submission script for Lucia — chapter baseline replication.
# 30 configs: 5 methods × 2 weightings × 3 datasets.
# Each method × weighting × dataset combination is one run.
#
#SBATCH --job-name=slp_hydra_replication
#SBATCH --time=12:00:00
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#SBATCH --array=0-1
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./out/slp_replication/%A_%a.out

module purge
module load EasyBuild/2025a
module load CUDA/12.8.0
module load Python/3.13.1-GCCcore-14.2.0

conda activate slp

config_files=(
#  "../../configs/hydra/replication/lsfb/1.actionness_with_weights.yaml"
#  "../../configs/hydra/replication/lsfb/2.actionness_without_weights.yaml"
  "../../configs/hydra/replication/lsfb/3.actionness_with_offsets_and_weights.yaml"
  "../../configs/hydra/replication/lsfb/4.actionness_with_offsets_without_weights.yaml"
)

config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Config file:  $config_file"
echo "Job start at $(date)"
python ../../scripts/segmentation/train.py --config-path="$config_file"
echo "Job end at $(date)"
