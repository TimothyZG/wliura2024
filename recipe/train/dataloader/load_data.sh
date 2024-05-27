#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=resnet-recipes/dataloader/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
python resnet-recipes/dataloader/dataloaders.py -d "DTD" -n 8