#!/bin/bash
#
#SBATCH --job-name=slp_hydra_vit
#SBATCH --time=24:00:00
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=32768
#SBATCH --partition=gpu
#SBATCH --array=0-5
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./out/vit_warm_20/%A_%a.out

source ~/miniconda3/etc/profile.d/conda.sh

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
  "../../configs/hydra/vit/vit_base.yaml"
  "../../configs/hydra/vit/vit_full_attn.yaml"
  "../../configs/hydra/vit/vit_xsmall.yaml"
  "../../configs/hydra/vit/vit_small.yaml"
  "../../configs/hydra/vit/vit_medium.yaml"
  "../../configs/hydra/vit/vit_large.yaml"
#  "../../configs/hydra/vit/vit_xlarge.yaml"
)

config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Config file:  $config_file"
echo "Job start at $(date)"
python ../../scripts/segmentation/train.py --config-path="$config_file"
echo "Job end at $(date)"
