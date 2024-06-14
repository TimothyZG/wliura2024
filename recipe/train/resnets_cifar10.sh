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
python recipe/train/train_resnet.py -d CIFAR10 -m Resnet18 -pn CIFAR10 -o ADAM -n 4 -e 40 -g 0.97 -r $SLURM_TMPDIR/data --lr 0.0015
python recipe/train/train_resnet.py -d CIFAR10 -m Resnet50 -pn CIFAR10 -o ADAM -n 4 -e 50 -g 0.97 -r $SLURM_TMPDIR/data --lr 0.0015
python recipe/train/train_resnet.py -d CIFAR10 -m Resnet101 -pn CIFAR10 -o ADAM -n 4 -e 60 -g 0.97 -r $SLURM_TMPDIR/data --lr 0.0015