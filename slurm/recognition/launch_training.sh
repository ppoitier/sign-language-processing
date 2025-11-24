#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=islr
#SBATCH --time=12:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#SBATCH --array=0-4
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./out/slp_%A_%a.out

module purge
module load EasyBuild/2024a
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0
module load FFmpeg/7.0.2-GCCcore-13.3.0

# Activate Python virtual env
source /gpfs/home/acad/unamur-fac_info/ppoitier/envs/slp/bin/activate

config_files=(
  "../../configs/islr/lsfb/pose2rgb/r18/lsfb500_r18_base.yaml"
  "../../configs/islr/lsfb/pose2rgb/r50/lsfb500_r50_base.yaml"
  "../../configs/islr/lsfb/pose2rgb/r152/lsfb500_r152_base.yaml"
  "../../configs/islr/lsfb/pose2rgb/pose-vit/lsfb500_posevit_jerome.yaml"
  "../../configs/islr/lsfb/pose2rgb/spoter/lsfb500_spoter_base.yaml"
)

config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Job start at $(date)"
python ../../slp/tasks/isolated_recognition/launch_training.py --config-path="$config_file"
echo "Job end at $(date)"