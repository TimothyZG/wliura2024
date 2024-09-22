#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
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

# Verify the extraction
ls $DATA_DIR  # This should now list the contents directly

# python recipe/train-iwc/iwcam.py --suffix 1 --config-path "recipe/train-iwc/config-train-18.yaml" --data-path "$DATA_DIR"
# python recipe/train-iwc/iwcam.py --suffix 2 --config-path "recipe/train-iwc/config-train-18.yaml" --data-path "$DATA_DIR"
# python recipe/train-iwc/iwcam.py --suffix 3 --config-path "recipe/train-iwc/config-train-18.yaml" --data-path "$DATA_DIR"

# python recipe/train-iwc/iwcam.py --suffix 5 --config-path "recipe/train-iwc/config-train-34.yaml" --data-path "$DATA_DIR"
# python recipe/train-iwc/iwcam.py --suffix 6 --config-path "recipe/train-iwc/config-train-34.yaml" --data-path "$DATA_DIR"
# python recipe/train-iwc/iwcam.py --suffix 7 --config-path "recipe/train-iwc/config-train-34.yaml" --data-path "$DATA_DIR"

# python recipe/train-iwc/iwcam.py --suffix 1 --config-path "recipe/train-iwc/config-train-50.yaml" --data-path "$DATA_DIR"
# python recipe/train-iwc/iwcam.py --suffix 2 --config-path "recipe/train-iwc/config-train-50.yaml" --data-path "$DATA_DIR"
# python recipe/train-iwc/iwcam.py --suffix 3 --config-path "recipe/train-iwc/config-train-50.yaml" --data-path "$DATA_DIR"

# python recipe/train-iwc/iwcam.py --suffix 3 --config-path "recipe/train-iwc/config-train-101.yaml" --data-path "$DATA_DIR"
# python recipe/train-iwc/iwcam.py --suffix 4 --config-path "recipe/train-iwc/config-train-101.yaml" --data-path "$DATA_DIR"
# python recipe/train-iwc/iwcam.py --suffix 5 --config-path "recipe/train-iwc/config-train-101.yaml" --data-path "$DATA_DIR"

# python recipe/train-iwc/iwcam.py --suffix 1 --config-path "recipe/train-iwc/config-train-152.yaml" --data-path "$DATA_DIR"
python recipe/train-iwc/iwcam.py --suffix 2 --config-path "recipe/train-iwc/config-train-152.yaml" --data-path "$DATA_DIR"


# python recipe/train-iwc/iwcam.py --suffix 3 --config-path "recipe/train-iwc/config-train-152.yaml" --data-path "$DATA_DIR"
# python recipe/train-iwc/iwcam.py --suffix 4 --config-path "recipe/train-iwc/config-train-152.yaml" --data-path "$DATA_DIR"
# python recipe/train-iwc/iwcam.py --suffix 5 --config-path "recipe/train-iwc/config-train-152.yaml" --data-path "$DATA_DIR"