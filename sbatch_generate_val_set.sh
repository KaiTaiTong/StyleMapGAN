#!/bin/bash
#SBATCH --job-name=gen_valset
#SBATCH --account=def-panos
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:v100l:1

source ~/scratch/StyleMapGAN/env/StyleMapGAN/bin/activate
python generate_local_editing_dataset.py --lmdb_file "data/celeba_hq/LMDB_val" --num_samples_per_class 625

echo "End"