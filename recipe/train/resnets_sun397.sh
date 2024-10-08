#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=16
#SBATCH --time=15:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/train/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
python recipe/train/train_resnet.py -d SUN397 -m Resnet18 -pn SUN397 -o ADAM -n 16 -e 60 -g 0.98 -r $SLURM_TMPDIR/data -ss 12 --lr 0.005 -lrs NONE --batch_size 256
python recipe/train/train_resnet.py -d SUN397 -m Resnet50 -pn SUN397 -o ADAM -n 16 -e 70 -g 0.98 -r $SLURM_TMPDIR/data -ss 14 --lr 0.005 -lrs NONE --batch_size 256
python recipe/train/train_resnet.py -d SUN397 -m Resnet101 -pn SUN397 -o ADAM -n 16 -e 80 -g 0.98 -r $SLURM_TMPDIR/data -ss 16 --lr 0.005 -lrs NONE --batch_size 256