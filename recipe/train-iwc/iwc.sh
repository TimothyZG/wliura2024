#!/bin/bash
#SBATCH --mem=64000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/train-iwc/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate

python recipe/train-iwc/iwcam.py --suffix 0 --config-path "recipe/train-iwc/config-test.yaml"