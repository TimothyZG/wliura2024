#!/bin/bash
#SBATCH --mem=64000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/finetune/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate


# python recipe/finetune/iwcam.py -d iWildCam -m Resnet18 -pn iWildCam-ss -w 16 -e 12 -dr Data --lr 3e-4 -nle 3 -s 0 -bs 32 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet18 -pn iWildCam-ss -w 16 -e 12 -dr Data --lr 3e-4 -nle 3 -s 1 -bs 32 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet18 -pn iWildCam-ss -w 16 -e 12 -dr Data --lr 3e-4 -nle 3 -s 2 -bs 32 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet18 -pn iWildCam-ss -w 16 -e 12 -dr Data --lr 3e-4 -nle 3 -s 3 -bs 32 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet18 -pn iWildCam-ss -w 16 -e 12 -dr Data --lr 3e-4 -nle 3 -s 4 -bs 32 -r 448 -lrs cosine -wd 0

# python recipe/finetune/iwcam.py -d iWildCam -m Resnet18 -pn iWildCam-ss -w 16 -e 15 -dr Data --lr 3e-4 -nle 3 -s 5 -bs 128 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet18 -pn iWildCam-ss -w 16 -e 15 -dr Data --lr 3e-4 -nle 3 -s 6 -bs 128 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet18 -pn iWildCam-ss -w 16 -e 15 -dr Data --lr 3e-4 -nle 3 -s 7 -bs 128 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet18 -pn iWildCam-ss -w 16 -e 15 -dr Data --lr 3e-4 -nle 3 -s 8 -bs 128 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet18 -pn iWildCam-ss -w 16 -e 15 -dr Data --lr 3e-4 -nle 3 -s 9 -bs 128 -r 448 -lrs cosine -wd 0

# python recipe/finetune/iwcam.py -d iWildCam -m Resnet34 -pn iWildCam-ss -w 16 -e 15 -dr Data --lr 3e-4 -nle 5 -s 0 -bs 32 -r 448 -lrs cosine -wd 0
python recipe/finetune/iwcam.py -d iWildCam -m Resnet34 -pn iWildCam-ss -w 16 -e 15 -dr Data --lr 3e-4 -nle 5 -s 1 -bs 32 -r 448 -lrs cosine -wd 0

# python recipe/finetune/iwcam.py -d iWildCam -m Resnet50 -pn iWildCam-ss -w 16 -e 15 -dr Data --lr 3e-4 -nle 3 -s 0 -bs 32 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet50 -pn iWildCam-ss -w 16 -e 15 -dr Data --lr 3e-4 -nle 3 -s 2 -bs 32 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet50 -pn iWildCam-ss -w 16 -e 20 -dr Data --lr 3e-4 -nle 5 -s 2 -bs 256 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet50 -pn iWildCam-ss -w 16 -e 20 -dr Data --lr 3e-4 -nle 5 -s 3 -bs 128 -r 448 -lrs cosine -wd 1e-5

# python recipe/finetune/iwcam.py -d iWildCam -m Resnet101 -pn iWildCam-ss -w 16 -e 15 -dr Data --lr 3e-4 -nle 3 -s 0 -bs 32 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet101 -pn iWildCam-ss -w 16 -e 15 -dr Data --lr 3e-4 -nle 3 -s 1 -bs 32 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet101 -pn iWildCam-ss -w 16 -e 20 -dr Data --lr 3e-4 -nle 5 -s 2 -bs 128 -r 448 -lrs cosine -wd 0

# python recipe/finetune/iwcam.py -d iWildCam -m Resnet152 -pn iWildCam-ss -w 16 -e 25 -dr Data --lr 3e-4 -nle 5 -s 0 -bs 32 -r 448 -lrs cosine -wd 0
# python recipe/finetune/iwcam.py -d iWildCam -m Resnet152 -pn iWildCam-ss -w 16 -e 25 -dr Data --lr 3e-4 -nle 5 -s 1 -bs 32 -r 448 -lrs cosine -wd 0
