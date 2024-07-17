#!/bin/bash
#SBATCH --mem=64000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/finetune/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
python recipe/finetune/main.py -d CIFAR10 -m Resnet18 -pn CIFAR10-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d CIFAR10 -m Resnet50 -pn CIFAR10-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d CIFAR10 -m Resnet101 -pn CIFAR10-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d CIFAR10 -m EffNet_S -pn CIFAR10-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d CIFAR10 -m EffNet_M -pn CIFAR10-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d CIFAR10 -m EffNet_L -pn CIFAR10-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
# python recipe/finetune/main.py -d CIFAR10 -m ViT_B_16 -pn CIFAR10-ms -w 16 -e 10 -dr $SLURM_TMPDIR/data --lr 0.001 -r 384 -bs 128
# python recipe/finetune/main.py -d CIFAR10 -m ViT_L_16 -pn CIFAR10-ms -w 16 -e 10 -dr $SLURM_TMPDIR/data --lr 0.001 -r 512 -bs 128
# python recipe/finetune/main.py -d CIFAR10 -m ViT_H_14 -pn CIFAR10-ms -w 16 -e 10 -dr $SLURM_TMPDIR/data --lr 0.001 -r 518 -bs 128