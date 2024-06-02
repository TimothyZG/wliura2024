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
python recipe/train/train_resnet.py -d SUN397 -m Resnet18 -pn SUN397 -o ADAM -n 4 -e 30 -g 0.95 -r $SLURM_TMPDIR/data
python recipe/train/train_resnet.py -d SUN397 -m Resnet50 -pn SUN397 -o ADAM -n 4 -e 30 -g 0.95 -r $SLURM_TMPDIR/data
python recipe/train/train_resnet.py -d SUN397 -m Resnet101 -pn SUN397 -o ADAM -n 4 -e 30 -g 0.95 -r $SLURM_TMPDIR/data