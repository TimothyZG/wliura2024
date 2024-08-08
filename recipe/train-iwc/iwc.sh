#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/train-iwc/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate

# Prepare data
mkdir -p $SLURM_TMPDIR/data/iwildcam_v2.0
tar xf Data/iwildcam_v2.0.tar -C $SLURM_TMPDIR/data/iwildcam_v2.0

python recipe/train-iwc/iwcam.py --suffix 0 --config-path "recipe/train-iwc/config-test.yaml"