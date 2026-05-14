#!/bin/bash
# Submission script for Lucia — chapter baseline replication.
# 30 configs: 5 methods × 2 weightings × 3 datasets.
# Each method × weighting × dataset combination is one run.
#
#SBATCH --job-name=sls_comparison
#SBATCH --time=06:00:00
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#SBATCH --array=0-13
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./out/1-comparison/%A_%a.out

module purge
module load EasyBuild/2025a
module load CUDA/12.8.0
#module load Python/3.13.1-GCCcore-14.2.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slp
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Should show conda's libstdc++, not /lib64/
ldconfig -p | grep libstdc++
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29

which python
python --version

config_files=(
#  "../../configs/segmentation/1-comparison/lsfb/io.yaml"
#  "../../configs/segmentation/1-comparison/lsfb/io_weights.yaml"
#
#  "../../configs/segmentation/1-comparison/lsfb/off.yaml"
#  "../../configs/segmentation/1-comparison/lsfb/off_weights.yaml"
#
#  "../../configs/segmentation/1-comparison/lsfb/bio.yaml"
#  "../../configs/segmentation/1-comparison/lsfb/bio_weights.yaml"
#
  "../../configs/segmentation/1-comparison/lsfb/bound_io.yaml"
  "../../configs/segmentation/1-comparison/lsfb/bound_io_weights.yaml"

  "../../configs/segmentation/1-comparison/lsfb/bound_off.yaml"
  "../../configs/segmentation/1-comparison/lsfb/bound_off_weights.yaml"

  "../../configs/segmentation/1-comparison/dgs/io.yaml"
  "../../configs/segmentation/1-comparison/dgs/io_weights.yaml"

  "../../configs/segmentation/1-comparison/dgs/off.yaml"
  "../../configs/segmentation/1-comparison/dgs/off_weights.yaml"

  "../../configs/segmentation/1-comparison/dgs/bio.yaml"
  "../../configs/segmentation/1-comparison/dgs/bio_weights.yaml"

  "../../configs/segmentation/1-comparison/dgs/bound_io.yaml"
  "../../configs/segmentation/1-comparison/dgs/bound_io_weights.yaml"

  "../../configs/segmentation/1-comparison/dgs/bound_off.yaml"
  "../../configs/segmentation/1-comparison/dgs/bound_off_weights.yaml"
)

config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Config file:  $config_file"
echo "Job start at $(date)"
python ../../scripts/segmentation/train.py --config-path="$config_file"
echo "Job end at $(date)"
