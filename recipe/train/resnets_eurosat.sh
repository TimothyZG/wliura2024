#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/train/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
python recipe/train/train_resnet.py -d EuroSAT -m Resnet18 -pn EuroSAT -o ADAM -n 4
python recipe/train/train_resnet.py -d EuroSAT -m Resnet50 -pn EuroSAT -o ADAM -n 4
python recipe/train/train_resnet.py -d EuroSAT -m Resnet101 -pn EuroSAT -o ADAM -n 4