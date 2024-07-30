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

python recipe/finetune/main.py -d DTD -m Resnet18 -pn DTD-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d DTD -m Resnet50 -pn DTD-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d DTD -m Resnet101 -pn DTD-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d DTD -m EffNet_S -pn DTD-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d DTD -m EffNet_M -pn DTD-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
python recipe/finetune/main.py -d DTD -m EffNet_L -pn DTD-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 