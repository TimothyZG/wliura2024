#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/train-iwc/output/slurm-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate

# Prepare data
export DATA_DIR=$SLURM_TMPDIR/data/iwildcam_v2.0
mkdir -p $DATA_DIR
tar xf Data/iwildcam_v2.0.tar -C $DATA_DIR --strip-components=1

# python recipe/train-iwc/train-save-along-the-way.py --suffix 1 --config-path "recipe/train-iwc/config-train-18-patw.yaml" --data-path "$DATA_DIR"
# python recipe/train-iwc/train-save-along-the-way.py --suffix 2 --config-path "recipe/train-iwc/config-train-18-patw.yaml" --data-path "$DATA_DIR"
python recipe/train-iwc/train-save-along-the-way.py --suffix 3 --config-path "recipe/train-iwc/config-train-18-patw.yaml" --data-path "$DATA_DIR"
