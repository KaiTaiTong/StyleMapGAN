import torch
from torch import nn
from training.model import Model, load_model
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils, transforms
import os
import csv
import pickle
import random
from tqdm import tqdm
import numpy as np
from training.dataset import MultiResolutionDataset, GTMaskDataset
import matplotlib.pyplot as plt
import warnings
import argparse
from PIL import Image

LOCAL_EDITING_PART_CHOICES = [
    "nose",
    "hair",
    "background",
    "eye",
    "eyebrow",
    "lip",
    "neck",
    "cloth",
    "skin",
    "ear",
]
PARTS_INDEX = {
    "background": [0],
    "skin": [1],
    "eyebrow": [6, 7],
    "eye": [3, 4, 5],
    "ear": [8, 9, 15],
    "nose": [2],
    "lip": [10, 11, 12],
    "neck": [16, 17],
    "cloth": [18],
    "hair": [13, 14],
}  # indices for each parts in the mask
DATASET_LIST = ['celeba_hq', 'afhq']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
])


def get_src_ref_pair(total_num_samples, num_pairs):
    """
    Randomly generate num_pairs src and ref indices pair given total number of samples. No 
    duplicated indices could be both src and ref at the same time, and no duplicated pairs 
    """
    # Do random selection without repeatation
    selection = random.sample(
        range(total_num_samples * (total_num_samples - 1)), num_pairs)
    indices1 = [i // (total_num_samples - 1) for i in selection]
    indices2 = [
        i % (total_num_samples - 1) + 1 if (i % (total_num_samples - 1)) >=
        (i // (total_num_samples - 1)) else i % (total_num_samples - 1)
        for i in selection
    ]
    return indices1, indices2


def save_src_ref_pair(output_file, indices1, indices2):
    """
    Save src and ref pairs as csv, based on img indices in lmdb dataset
    """
    assert (len(indices1) == len(indices2)
            ), "indices1 and indices2 must have same size"
    with open(output_file, 'w') as file:
        file.write('mixed_index, src_index, ref_index\n')
        for i, _ in enumerate(indices1):
            file.write(f"{i}, {indices1[i]}, {indices2[i]}\n")
    print("file completed")


def save_image(img, path, normalize=True, range=(-1, 1)):
    utils.save_image(
        img,
        path,
        normalize=normalize,
        range=range,
    )


def save_images(imgs, paths, normalize=True, range=(-1, 1)):
    for img, path in zip(imgs, paths):
        save_image(img, path, normalize=normalize, range=range)


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def generate_reconst_imgs(model, loader, save_image_dir):
    """
    Generate reconstructed imgs from g_ema
    """
    for i, (real_img, mask, _) in enumerate(tqdm(loader)):
        real_img = real_img.to(device)
        recon_image = model(real_img, "reconstruction")
        save_images([recon_image], [f"{save_image_dir}/{i}_recon.png"])


def save_label_mask(loader, save_image_dir):
    """
    Save label masks for the recons imgs
    """
    for i, (real_img, mask, _) in enumerate(tqdm(loader)):
        mask = mask.to(device)
        im = Image.fromarray(mask.to('cpu', torch.uint8).numpy().squeeze())
        im.save(f"{save_image_dir}/{i}_mask.png")


def generate_gender_table(loader, save_table_dir):
    """
    Generate gender look up table for reconstructed imgs. Male: 0; Female: 1 
    """
    gender_table = {}
    for i, (_, _, annotation) in enumerate(tqdm(loader)):
        gender_table[i] = 1 if annotation['Male'] == -1 else 0

    with open(os.path.join(save_table_dir, 'gender_table.pkl'),
              'wb') as handle:
        pickle.dump(gender_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_local_edited_imgs(model, loader, save_image_dir,
                               num_fake_samples_per_class, fake_real_split):
    """
    Generate local edited imgs (fakes) and their editing masks, along with a lookup table 
    file that indicates indices of src_ref pairs of fakes and unused reals

    Fakes: Generated from random pick pairs of src and ref in 0:(total_samples*fake_real_split)
    Reals: Unused re-constructed imgs in (total_samples*fake_real_split):-1

    Note:
        For each local edited parts, the selection of src_ref pairs is re-generated randomly from 
        the same fakes pool
    """
    for local_editing_part in LOCAL_EDITING_PART_CHOICES:
        save_image_child_dir = os.path.join(save_image_dir, local_editing_part)
        for kind in [
                "editing_mask",  # binary mask for local editing region
                "synthesized_image"  # local edited img
        ]:
            os.makedirs(os.path.join(save_image_child_dir, kind),
                        exist_ok=True)

        # Local editing
        # We don't want to hand pick the pairing based on similarity, because
        # we want the prediction model to be trained on a mixture of good and
        # bad samples

        # Fakes are generated via random pick pairs of src and ref in 0:fakes_pool_size
        # Reals are unused re-constructed imgs in fakes_pool_size:-1
        fakes_pool_size = int(n_sample * fake_real_split)
        reals_pool_size = n_sample - fakes_pool_size
        indices1, indices2 = get_src_ref_pair(fakes_pool_size,
                                              num_fake_samples_per_class)

        with torch.no_grad():
            for loop_i, (index1,
                         index2) in tqdm(enumerate(zip(indices1, indices2)),
                                         total=len(indices1)):

                src_img, mask1_logit, _ = loader.dataset[index1]
                ref_img, mask2_logit, _ = loader.dataset[index2]

                src_img = src_img.to(device)
                ref_img = ref_img.to(device)
                mask1_logit = mask1_logit.to(device)
                mask2_logit = mask2_logit.to(device)

                src_img, ref_img = src_img[None, :], ref_img[None, :]
                latents1, latents2 = model(src_img, "projection").squeeze(0), \
                                    model(ref_img, "projection").squeeze(0)

                mask1 = -torch.ones(mask1_logit.shape).to(
                    device)  # initialize with -1
                mask2 = -torch.ones(mask2_logit.shape).to(
                    device)  # initialize with -1

                for label_i in PARTS_INDEX[local_editing_part]:
                    mask1[(mask1_logit == label_i) == True] = 1
                    mask2[(mask2_logit == label_i) == True] = 1

                mask = mask1 + mask2
                mask = mask.float()

                mixed_image, recon_img_src, recon_img_ref = model(
                    (latents1, latents2, mask), "local_editing")

                # Convert to binary mask
                mask[mask < -1] = -1
                mask[mask > -1] = 1

                save_images(
                    [
                        mask,
                        mixed_image[0],
                    ],
                    [
                        f"{save_image_child_dir}/editing_mask/{loop_i}.png",
                        f"{save_image_child_dir}/synthesized_image/{loop_i}.png"
                    ],
                )

                save_src_ref_pair(
                    os.path.join(save_image_child_dir, 'src_ref_pair.txt'),
                    indices1, indices2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",
                        type=str,
                        default="./expr/checkpoints/celeba_hq_256_8x8.pt")
    parser.add_argument("--lmdb_file", type=str, default="data/celeba_hq/LMDB")
    parser.add_argument("--num_fake_samples_per_class", type=int, default=5)
    parser.add_argument("--fake_real_split", type=float, default=0.5)
    # parser.add_argument("--reconstructed_img_path",
    #                     type=str,
    #                     default="./dataset/reconstructed_imgs")
    # parser.add_argument(
    #     "--label_mask_path",
    #     type=str,
    #     default=
    #     "/project/6003167/alantkt/datasets/local_editing_dataset/segmentation_imgs"
    # )
    # parser.add_argument("--output_dataset_path",
    #                     type=str,
    #                     default="./dataset/local_edited_imgs")  # None
    parser.add_argument(
        "--gender_path",
        type=str,
        default="/project/6003167/alantkt/datasets/local_editing_dataset/")
    # ========= Uncomment this to remove generation tasks =========
    parser.add_argument("--reconstructed_img_path", type=str, default=None)
    parser.add_argument("--label_mask_path", type=str, default=None)
    parser.add_argument("--output_dataset_path", type=str, default=None)
    # parser.add_argument("--gender_path", type=str, default=None)

    args = parser.parse_args()

    ckpt_path = args.ckpt
    lmdb_file = args.lmdb_file  # input images in mdb file
    num_fake_samples_per_class = args.num_fake_samples_per_class
    fake_real_split = args.fake_real_split
    batch = 1
    num_workers = 0  # fixed 0

    device = "cuda"

    # Load model parameters
    model = load_model(ckpt_path, device=device)
    train_args = model.train_args
    assert (train_args.dataset == "celeba_hq"
            ), "train_args.dataset must be celeba_hq"

    # CelebA dataset contains an RGB img, and a classification mask pair
    dataset = GTMaskDataset(lmdb_file, transform, train_args.size)

    # Prepare Dataloader
    n_sample = len(dataset)
    sampler = data_sampler(dataset, shuffle=False)

    loader = data.DataLoader(
        dataset,
        batch,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # generated images should match with n sample
    if n_sample % batch == 0:
        assert len(loader) == n_sample // batch
    else:
        assert len(loader) == n_sample // batch + 1

    # Generate generator-reconstructed imgs
    if args.reconstructed_img_path is not None:
        print("Generate StyleMapGAN reconstructed dataset")
        os.makedirs(args.reconstructed_img_path, exist_ok=True)

        generate_reconst_imgs(model, loader, args.reconstructed_img_path)

    # Get and save segmantation masks for reconstructed dataset
    if args.label_mask_path is not None:
        print("Save segmentation masks for reconstructed dataset")
        os.makedirs(args.label_mask_path, exist_ok=True)

        save_label_mask(loader, args.label_mask_path)

    # Generate local edited dataset
    if args.output_dataset_path is not None:
        print("Generate local edited dataset")
        os.makedirs(args.output_dataset_path, exist_ok=True)

        generate_local_edited_imgs(model, loader, args.output_dataset_path,
                                   num_fake_samples_per_class, fake_real_split)

    # Generate gender lookup table
    if args.gender_path is not None:
        print("Generate gender path")

        generate_gender_table(loader, args.gender_path)
