#!/bin/bash
#SBATCH --job-name=gen_trainset
#SBATCH --account=def-panos
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:v100l:1

source ~/scratch/StyleMapGAN/env/StyleMapGAN/bin/activate
python generate_local_editing_dataset.py --lmdb_file "data/celeba_hq/LMDB_train" --num_samples_per_class 5000

echo "End"