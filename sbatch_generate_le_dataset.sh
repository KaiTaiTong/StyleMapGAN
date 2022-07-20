#!/bin/bash
#SBATCH --job-name=pre_process
#SBATCH --account=def-panos
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100l:1

source ~/scratch/StyleMapGAN/env/StyleMapGAN/bin/activate
# python preprocessor/prepare_data.py --out data/celeba_hq/LMDB_train --size "256,1024" data/celeba_hq/raw_images/train
python preprocessor/prepare_data.py --out data/celeba_hq/LMDB_val --size "256,1024" data/celeba_hq/raw_images/val

echo "End"