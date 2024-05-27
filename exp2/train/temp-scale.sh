#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
pwd
ls
cd models
ls
cd ..
python exp2/train/ts.py