#!/bin/bash
#SBATCH --mem=64000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/finetune/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
python recipe/finetune/main.py -d MNIST -m Resnet18 -pn MNIST-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d MNIST -m Resnet50 -pn MNIST-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d MNIST -m Resnet101 -pn MNIST-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d MNIST -m EffNet_S -pn MNIST-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d MNIST -m EffNet_M -pn MNIST-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d MNIST -m EffNet_L -pn MNIST-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 