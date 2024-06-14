#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/train/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
export CUDA_LAUNCH_BLOCKING=1
python recipe/train/train_resnet.py -d Caltech256 -m Resnet18 -pn Caltech256 -o ADAM -n 4 -e 60 -g 0.98 -r $SLURM_TMPDIR/data
export CUDA_LAUNCH_BLOCKING=1
python recipe/train/train_resnet.py -d Caltech256 -m Resnet50 -pn Caltech256 -o ADAM -n 4 -e 80 -g 0.98 -r $SLURM_TMPDIR/data
export CUDA_LAUNCH_BLOCKING=1
python recipe/train/train_resnet.py -d Caltech256 -m Resnet101 -pn Caltech256 -o ADAM -n 4 -e 100 -g 0.98 -r $SLURM_TMPDIR/data