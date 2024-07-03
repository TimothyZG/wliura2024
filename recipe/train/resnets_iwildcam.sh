#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=30:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/train/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
# python recipe/train/train_resnet.py -d iWildCam -m Resnet18 -pn iWildCam -o ADAM -n 4 -e 40 -r $SLURM_TMPDIR/data -g 0.93  -r $SLURM_TMPDIR/data --lr 0.01
# python recipe/train/train_resnet.py -d iWildCam -m Resnet50 -pn iWildCam -o ADAM -n 4 -e 50 -r $SLURM_TMPDIR/data -g 0.93  -r $SLURM_TMPDIR/data --lr 0.01
# python recipe/train/train_resnet.py -d iWildCam -m Resnet101 -pn iWildCam -o ADAM -n 4 -e 60 -r $SLURM_TMPDIR/data -g 0.94  -r $SLURM_TMPDIR/data --lr 0.01

# python recipe/train/train_resnet.py -d iWildCam -m Resnet18 -pn iWildCam -o ADAM -n 4 -e 50  -g 0.1 -r $SLURM_TMPDIR/data -ss 20 --lr 0.01 -lrs STEP
# python recipe/train/train_resnet.py -d iWildCam -m Resnet50 -pn iWildCam -o ADAM -n 4 -e 60  -g 0.1 -r $SLURM_TMPDIR/data -ss 20 --lr 0.01 -lrs STEP
# python recipe/train/train_resnet.py -d iWildCam -m Resnet101 -pn iWildCam -o ADAM -n 4 -e 70  -g 0.1 -r $SLURM_TMPDIR/data -ss 20 --lr 0.01 -lrs STEP

CUDA_LAUNCH_BLOCKING=1.
# python recipe/train/train_resnet.py -d iWildCam -m Resnet18 -pn iWildCam -o ADAM -n 16 -e 80 -g 0.5 -r $SLURM_TMPDIR/data -ss 15 --lr 0.005 -lrs NONE --batch_size 128
# python recipe/train/train_resnet.py -d iWildCam -m Resnet50 -pn iWildCam -o ADAM -n 16 -e 100  -g 0.95 -r $SLURM_TMPDIR/data -ss 20 --lr 0.005 -lrs NONE --batch_size 128
python recipe/train/train_resnet.py -d iWildCam -m Resnet101 -pn iWildCam -o ADAM -n 16 -e 120  -g 0.96 -r $SLURM_TMPDIR/data -ss 24 --lr 0.005 -lrs NONE --batch_size 512