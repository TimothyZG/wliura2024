#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
CUDA_LAUNCH_BLOCKING=1 python exp2/resnet18.py