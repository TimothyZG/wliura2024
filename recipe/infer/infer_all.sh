#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/infer/output/slurm-%j.out

module purge
module load python/3.10 scipy-stack
source ~/py310/bin/activate

python recipe/infer/infer_resnet.py -d DTD -a Resnet18 -n 4 -m Models/Resnet18-DTD-SGD.pth #  -p Pred/pred_res18.csv -t Pred/target_dtd.csv
python recipe/infer/infer_resnet.py -d DTD -a Resnet50 -n 4 -m Models/Resnet50-DTD-SGD.pth # -p Pred/pred_res50.csv -t Pred/target_dtd.csv
python recipe/infer/infer_resnet.py -d DTD -a Resnet101 -n 4 -m Models/Resnet101-DTD-SGD.pth # -p Pred/pred_res101.csv -t Pred/target_dtd.csv

python recipe/infer/infer_resnet.py -d EuroSAT -a Resnet18 -n 4 -m Models/Resnet18-EuroSAT-ADAM.pth # -p Pred/pred_eurosat_res18.csv -t Pred/target_dtd.csv
python recipe/infer/infer_resnet.py -d EuroSAT -a Resnet50 -n 4 -m Models/Resnet50-EuroSAT-ADAM.pth # -p Pred/pred_eurosat_res50.csv -t Pred/target_dtd.csv
python recipe/infer/infer_resnet.py -d EuroSAT -a Resnet101 -n 4 -m Models/Resnet101-EuroSAT-ADAM.pth # -p Pred/pred_eurosat_res101.csv -t Pred/target_dtd.csv

python recipe/infer/infer_resnet.py -d GTSRB -a Resnet18 -n 4 -m Models/Resnet18-GTSRB-ADAM.pth
python recipe/infer/infer_resnet.py -d GTSRB -a Resnet50 -n 4 -m Models/Resnet50-GTSRB-ADAM.pth
python recipe/infer/infer_resnet.py -d GTSRB -a Resnet101 -n 4 -m Models/Resnet101-GTSRB-ADAM.pth
