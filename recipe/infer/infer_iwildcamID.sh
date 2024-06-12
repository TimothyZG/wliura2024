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

python recipe/infer/infer_resnet.py -d iWildCamID -a Resnet18 -n 4 -m Models/Resnet18-iWildCam-ADAM.pth -r $SLURM_TMPDIR/data
python recipe/infer/infer_resnet.py -d iWildCamID -a Resnet50 -n 4 -m Models/Resnet50-iWildCam-ADAM.pth -r $SLURM_TMPDIR/data
python recipe/infer/infer_resnet.py -d iWildCamID -a Resnet101 -n 4 -m Models/Resnet101-iWildCam-ADAM.pth -r $SLURM_TMPDIR/data
