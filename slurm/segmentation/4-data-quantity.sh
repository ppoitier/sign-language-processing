#!/bin/bash
# Submission script for Lucia — data quantity sweep.
# 10 configs: 5 training-shard subsets (data_1..data_5) × 2 datasets.
#
#SBATCH --job-name=sls_data_quantity
#SBATCH --time=06:00:00
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#SBATCH --array=0-9
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./out/4-data-quantity/%A_%a.out

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
  "../../configs/segmentation/4-data-quantity/lsfb/data_1.yaml"
  "../../configs/segmentation/4-data-quantity/lsfb/data_2.yaml"
  "../../configs/segmentation/4-data-quantity/lsfb/data_3.yaml"
  "../../configs/segmentation/4-data-quantity/lsfb/data_4.yaml"
  "../../configs/segmentation/4-data-quantity/lsfb/data_5.yaml"

  "../../configs/segmentation/4-data-quantity/dgs/data_1.yaml"
  "../../configs/segmentation/4-data-quantity/dgs/data_2.yaml"
  "../../configs/segmentation/4-data-quantity/dgs/data_3.yaml"
  "../../configs/segmentation/4-data-quantity/dgs/data_4.yaml"
  "../../configs/segmentation/4-data-quantity/dgs/data_5.yaml"
)

config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Config file:  $config_file"
echo "Job start at $(date)"
python ../../scripts/segmentation/train.py --config-path="$config_file"
echo "Job end at $(date)"
