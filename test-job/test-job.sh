#!/bin/bash
#SBATCH --mem=1024M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:3:00    
#SBATCH --mail-user=<tzhou13@student.ubc.ca>
#SBATCH --mail-type=ALL
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
# python test-job/test-script.py
python test-job/wandb-script.py