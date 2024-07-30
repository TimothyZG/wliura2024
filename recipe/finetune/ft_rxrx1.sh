#!/bin/bash
#SBATCH --mem=64000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/finetune/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate


# python recipe/finetune/iwcam.py -d rxrx1 -m Resnet18 -pn rxrx1-ss -w 16 -e 90 -dr $SLURM_TMPDIR/data --lr 3e-4 -nle 1 -s 0 -bs 256 -r 224 -lrs NONE -wd 0
# python recipe/finetune/iwcam.py -d rxrx1 -m Resnet18 -pn rxrx1-ss -w 16 -e 90 -dr $SLURM_TMPDIR/data --lr 3e-4 -nle 1 -s 1 -bs 256 -r 224 -lrs NONE -wd 0
# python recipe/finetune/iwcam.py -d rxrx1 -m Resnet18 -pn rxrx1-ss -w 16 -e 90 -dr $SLURM_TMPDIR/data --lr 3e-4 -nle 1 -s 2 -bs 256 -r 224 -lrs NONE -wd 0
# python recipe/finetune/iwcam.py -d rxrx1 -m Resnet18 -pn rxrx1-ss -w 16 -e 90 -dr $SLURM_TMPDIR/data --lr 3e-4 -nle 1 -s 3 -bs 256 -r 224 -lrs NONE -wd 0
# python recipe/finetune/iwcam.py -d rxrx1 -m Resnet18 -pn rxrx1-ss -w 16 -e 90 -dr $SLURM_TMPDIR/data --lr 3e-4 -nle 1 -s 4 -bs 256 -r 224 -lrs NONE -wd 0
# python recipe/finetune/iwcam.py -d rxrx1 -m Resnet50 -pn rxrx1-ss -w 16 -e 90 -dr $SLURM_TMPDIR/data --lr 3e-4 -nle 1 -s 0 -bs 256 -r 224 -lrs NONE -wd 0
python recipe/finetune/iwcam.py -d rxrx1 -m Resnet101 -pn rxrx1-ss -w 16 -e 90 -dr $SLURM_TMPDIR/data --lr 3e-4 -nle 1 -s 0 -bs 128 -r 224 -lrs cosine -wd 0
