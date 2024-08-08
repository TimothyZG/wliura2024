#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/train-iwc/output/slurm-%j.out
module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate

# Prepare data
export DATA_DIR=$SLURM_TMPDIR/data/iwildcam_v2.0
mkdir -p $DATA_DIR
tar xf Data/iwildcam_v2.0.tar -C $DATA_DIR --strip-components=1

# Verify the extraction
ls $DATA_DIR  # This should now list the contents directly

python recipe/train-iwc/hyper-param-tune.py --config-path "recipe/train-iwc/config.yaml"  --data-path "$DATA_DIR"