#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
python exp2-inf/infer-test.py