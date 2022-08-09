#!/bin/bash
#SBATCH --job-name=gen_testset
#SBATCH --account=def-panos
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:v100l:1

source ~/scratch/StyleMapGAN/env/StyleMapGAN/bin/activate
python generate_local_editing_dataset.py --lmdb_file "data/celeba_hq/LMDB" --num_samples_per_class 3000 --reconstructed_img_path "/scratch/alantkt/local_editing_dataset/reconstructed_imgs" --output_dataset_path "/scratch/alantkt/local_editing_dataset/local_edited_imgs"

echo "End"