#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=slp
#SBATCH --time=08:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#SBATCH --array=0-23
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./out/slp_%A_%a.out

module purge
module load EasyBuild/2022a
module load Python/3.10.4-GCCcore-11.3.0

# Activate Python virtual env
source /gpfs/home/acad/unamur-fac_info/ppoitier/envs/dl/bin/activate

config_files=(
  "../configs/dgs/actionness.yaml"
  "../configs/dgs/actionness_with_offsets.yaml"
  "../configs/dgs/bio_tagging.yaml"
  "../configs/dgs/bio_tagging_without_weights.yaml"
  "../configs/dgs/boundaries.yaml"
  "../configs/dgs/boundaries_without_weights.yaml"
  "../configs/dgs/boundaries_with_offsets.yaml"
  "../configs/dgs/boundaries_with_offsets_without_weights.yaml"

  "../configs/lsfb/actionness.yaml"
  "../configs/lsfb/actionness_with_offsets.yaml"
  "../configs/lsfb/bio_tagging.yaml"
  "../configs/lsfb/bio_tagging_without_weights.yaml"
  "../configs/lsfb/boundaries.yaml"
  "../configs/lsfb/boundaries_without_weights.yaml"
  "../configs/lsfb/boundaries_with_offsets.yaml"
  "../configs/lsfb/boundaries_with_offsets_without_weights.yaml"

  "../configs/phoenix/actionness.yaml"
  "../configs/phoenix/actionness_with_offsets.yaml"
  "../configs/phoenix/bio_tagging.yaml"
  "../configs/phoenix/bio_tagging_without_weights.yaml"
  "../configs/phoenix/boundaries.yaml"
  "../configs/phoenix/boundaries_without_weights.yaml"
  "../configs/phoenix/boundaries_with_offsets.yaml"
  "../configs/phoenix/boundaries_with_offsets_without_weights.yaml"
)

config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Job start at $(date)"
python ../scripts/segmentation/train.py --config-path="$config_file"
echo "Job end at $(date)"