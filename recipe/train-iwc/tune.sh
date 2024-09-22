#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/train-iwc/output/tune-%j.out
module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate

# Prepare data
export DATA_DIR=$SLURM_TMPDIR/data/iwildcam_v2.0
mkdir -p $DATA_DIR
tar xf Data/iwildcam_v2.0.tar -C $DATA_DIR --strip-components=1

# Verify the extraction
ls $DATA_DIR

# python recipe/train-iwc/hyper-param-tune.py --config-path "recipe/train-iwc/config-tune-18.yaml"  --data-path "$DATA_DIR"
# python recipe/train-iwc/hyper-param-tune.py --config-path "recipe/train-iwc/config-tune-34.yaml"  --data-path "$DATA_DIR"
# python recipe/train-iwc/hyper-param-tune.py --config-path "recipe/train-iwc/config-tune-50.yaml"  --data-path "$DATA_DIR"
# python recipe/train-iwc/hyper-param-tune.py --config-path "recipe/train-iwc/config-tune-101.yaml"  --data-path "$DATA_DIR"
python recipe/train-iwc/hyper-param-tune.py --config-path "recipe/train-iwc/config-tune-152.yaml"  --data-path "$DATA_DIR"