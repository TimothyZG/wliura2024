#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/finetune/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate
# python recipe/finetune/main.py -d iWildCam -m Resnet18 -pn iWildCam-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
# python recipe/finetune/main.py -d iWildCam -m Resnet50 -pn iWildCam-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
# python recipe/finetune/main.py -d iWildCam -m Resnet101 -pn iWildCam-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 

# python recipe/finetune/main.py -d iWildCam -m EffNet_S -pn iWildCam-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
# python recipe/finetune/main.py -d iWildCam -m EffNet_M -pn iWildCam-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 
# python recipe/finetune/main.py -d iWildCam -m EffNet_L -pn iWildCam-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 

python recipe/finetune/main.py -d iWildCam -m ViT_B_16 -pn iWildCam-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 -r 384
python recipe/finetune/main.py -d iWildCam -m ViT_L_16 -pn iWildCam-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 -r 512
python recipe/finetune/main.py -d iWildCam -m ViT_H_14 -pn iWildCam-ms -w 16 -e 30 -dr $SLURM_TMPDIR/data --lr 0.001 -r 518