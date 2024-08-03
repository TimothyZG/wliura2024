#!/bin/bash
#SBATCH --mem=64000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/iwc/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate

python recipe/iwc/hyper-param-tune.py --config-path "recipe/iwc/config.yaml"