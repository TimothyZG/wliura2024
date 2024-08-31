#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=14:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/rxrx1/output/train-%j.out
module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate

# Prepare data
export DATA_DIR=$SLURM_TMPDIR/data/rxrx1_v1.0
mkdir -p $DATA_DIR
tar xf Data/rxrx1_v1.0/archive.tar.gz -C $DATA_DIR --strip-components=1

# Verify the extraction
ls $DATA_DIR  # This should now list the contents directly


# python recipe/rxrx1/train.py --suffix 0 --config-path "recipe/rxrx1/config-train-18.yaml" --data-path "$DATA_DIR"

# python recipe/rxrx1/train.py --suffix 0 --config-path "recipe/rxrx1/config-train-34.yaml" --data-path "$DATA_DIR"

python recipe/rxrx1/train.py --suffix 4 --config-path "recipe/rxrx1/config-train-50.yaml" --data-path "$DATA_DIR"

# python recipe/rxrx1/train.py --suffix 0 --config-path "recipe/rxrx1/config-train-101.yaml" --data-path "$DATA_DIR"

# python recipe/rxrx1/train.py --suffix 0 --config-path "recipe/rxrx1/config-train-152.yaml" --data-path "$DATA_DIR"
