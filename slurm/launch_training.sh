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
#SBATCH --array=0-29
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
  "../configs/dgs/1.actionness_with_weights.yaml"
  "../configs/dgs/2.actionness_without_weights.yaml"
  "../configs/dgs/3.actionness_with_offsets_and_weights.yaml"
  "../configs/dgs/4.actionness_with_offsets_without_weights.yaml"
  "../configs/dgs/5.bio_tagging_with_weights.yaml"
  "../configs/dgs/6.bio_tagging_without_weights.yaml"
  "../configs/dgs/7.boundaries_with_weights.yaml"
  "../configs/dgs/8.boundaries_without_weights.yaml"
  "../configs/dgs/9.boundaries_with_offsets_and_weights.yaml"
  "../configs/dgs/10.boundaries_with_offsets_without_weights.yaml"

  "../configs/lsfb/1.actionness_with_weights.yaml"
  "../configs/lsfb/2.actionness_without_weights.yaml"
  "../configs/lsfb/3.actionness_with_offsets_and_weights.yaml"
  "../configs/lsfb/4.actionness_with_offsets_without_weights.yaml"
  "../configs/lsfb/5.bio_tagging_with_weights.yaml"
  "../configs/lsfb/6.bio_tagging_without_weights.yaml"
  "../configs/lsfb/7.boundaries_with_weights.yaml"
  "../configs/lsfb/8.boundaries_without_weights.yaml"
  "../configs/lsfb/9.boundaries_with_offsets_and_weights.yaml"
  "../configs/lsfb/10.boundaries_with_offsets_without_weights.yaml"

  "../configs/phoenix/1.actionness_with_weights.yaml"
  "../configs/phoenix/2.actionness_without_weights.yaml"
  "../configs/phoenix/3.actionness_with_offsets_and_weights.yaml"
  "../configs/phoenix/4.actionness_with_offsets_without_weights.yaml"
  "../configs/phoenix/5.bio_tagging_with_weights.yaml"
  "../configs/phoenix/6.bio_tagging_without_weights.yaml"
  "../configs/phoenix/7.boundaries_with_weights.yaml"
  "../configs/phoenix/8.boundaries_without_weights.yaml"
  "../configs/phoenix/9.boundaries_with_offsets_and_weights.yaml"
  "../configs/phoenix/10.boundaries_with_offsets_without_weights.yaml"
)

config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Job start at $(date)"
python ../scripts/segmentation/train.py --config-path="$config_file"
echo "Job end at $(date)"