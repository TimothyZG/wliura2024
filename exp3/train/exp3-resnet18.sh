#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=output/slurm-%j.out
cd $SLURM_TMPDIR
git clone https://github.com/TimothyZG/wliura2024.git
cd ./wliura2024
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
python exp3/train/resnet18.py
cp $SLURM_TMPDIR/* /$project/wliura2024/exp3/outputs