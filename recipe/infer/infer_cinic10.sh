#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=06:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/infer/output/slurm-%j.out

module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate

python recipe/infer/infer_resnet.py -d CINIC10 -a Resnet18 -n 4 -m Models/Resnet18-CIFAR10.pth -r $SLURM_TMPDIR/data
python recipe/infer/infer_resnet.py -d CINIC10 -a Resnet50 -n 4 -m Models/Resnet50-CIFAR10.pth -r $SLURM_TMPDIR/data
python recipe/infer/infer_resnet.py -d CINIC10 -a Resnet101 -n 4 -m Models/Resnet101-CIFAR10.pth -r $SLURM_TMPDIR/data

python recipe/infer/infer_resnet.py -d CINIC10 -a EffNet_S -n 4 -m Models/EffNet_S-CIFAR10.pth -r $SLURM_TMPDIR/data
python recipe/infer/infer_resnet.py -d CINIC10 -a EffNet_M -n 4 -m Models/EffNet_M-CIFAR10.pth -r $SLURM_TMPDIR/data
python recipe/infer/infer_resnet.py -d CINIC10 -a EffNet_L -n 4 -m Models/EffNet_L-CIFAR10.pth -r $SLURM_TMPDIR/data

# python recipe/infer/infer_resnet.py -d CINIC10 -a ViT_B_16 -n 4 -m Models/ViT_B_16-CIFAR10.pth -r $SLURM_TMPDIR/data -rs 384
# python recipe/infer/infer_resnet.py -d CINIC10 -a ViT_L_16 -n 4 -m Models/ViT_L_16-CIFAR10.pth -r $SLURM_TMPDIR/data -rs 512
# python recipe/infer/infer_resnet.py -d CINIC10 -a ViT_L_16 -n 4 -m Models/ViT_L_16-CIFAR10.pth -r $SLURM_TMPDIR/data -rs518