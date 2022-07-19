from io import BytesIO

"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torch.utils import data
import numpy as np
import random
import re, os
from torchvision import transforms
import torch


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class GTMaskDataset(Dataset):
    def __init__(self, lmdb_folder, transform, resolution=256):

        # Get images dataset type
        if 'LMDB_train' in lmdb_folder:
            dataset_type = 'train'
        elif 'LMDB_test' in lmdb_folder:
            dataset_type = 'test'
        elif 'LMDB_val' in lmdb_folder:
            dataset_type = 'val'
        else:
            raise ValueError("Unknown lmdb dataset type")
        
        # dataset_folder is the parent directory of lmdb_folder
        dataset_folder = '/'.join(lmdb_folder.split('/')[:-1])

        self.env = lmdb.open(
            lmdb_folder,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError("Cannot open lmdb dataset", lmdb_folder)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        self.resolution = resolution
        self.transform = transform

        # convert filename to celeba_hq index
        CelebA_HQ_to_CelebA = (
            f"{dataset_folder}/local_editing/CelebA-HQ-to-CelebA-mapping.txt"
        )
        CelebA_to_CelebA_HQ_dict = {}

        # Path to original images and mask labels
        original_path = f"{dataset_folder}/raw_images/{dataset_type}/images"
        mask_label_path = f"{dataset_folder}/CelebAMask-HQ/CelebAMaskHQ-mask"

        # Fill CelebA_to_CelebA_HQ_dict dict that maps celeba dataset *.jpg -> index
        # for a total of 30000 samples
        with open(CelebA_HQ_to_CelebA, "r") as fp:
            read_line = fp.readline()
            attrs = re.sub(" +", " ", read_line).strip().split(" ")
            while True:
                read_line = fp.readline()

                if not read_line:
                    break

                idx, orig_idx, orig_file = (
                    re.sub(" +", " ", read_line).strip().split(" ")
                )

                CelebA_to_CelebA_HQ_dict[orig_file] = idx

        self.mask = []

        # LMDB file is generated in alphabetical order of the imgs file, thus we need to 
        # sort the listdir as well to make mask match the input imgs
        for filename in sorted(os.listdir(original_path)):
            CelebA_HQ_filename = CelebA_to_CelebA_HQ_dict[filename]
            CelebA_HQ_filename = CelebA_HQ_filename + ".png"
            self.mask.append(os.path.join(mask_label_path, CelebA_HQ_filename))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        mask = Image.open(self.mask[index])

        mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)
        mask = transforms.ToTensor()(mask)

        mask = mask.squeeze()
        mask *= 255
        mask = mask.long()

        assert mask.shape == (self.resolution, self.resolution)
        return img, mask


class DataSetFromDir(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = []

        for img in all_imgs:
            if ".png" in img:
                self.total_imgs.append(img)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


class DataSetTestLocalEditing(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform

        all_imgs = os.listdir(os.path.join(main_dir, "mask"))
        self.total_imgs = []

        for img in all_imgs:
            if ".png" in img:
                self.total_imgs.append(img)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        image_mask = self.transform(
            Image.open(
                os.path.join(self.main_dir, "mask", self.total_imgs[idx])
            ).convert("RGB")
        )
        image_reference = self.transform(
            Image.open(
                os.path.join(self.main_dir, "reference_image", self.total_imgs[idx])
            ).convert("RGB")
        )
        # image_reference_recon = self.transform(Image.open(os.path.join(self.main_dir, 'reference_image', self.total_imgs[idx].replace('.png', '_recon_img.png'))).convert("RGB"))

        image_source = self.transform(
            Image.open(
                os.path.join(self.main_dir, "source_image", self.total_imgs[idx])
            ).convert("RGB")
        )
        # image_source_recon = self.transform(Image.open(os.path.join(self.main_dir, 'source_image', self.total_imgs[idx].replace('.png', '_recon_img.png'))).convert("RGB"))

        image_synthesized = self.transform(
            Image.open(
                os.path.join(self.main_dir, "synthesized_image", self.total_imgs[idx])
            ).convert("RGB")
        )

        return image_mask, image_reference, image_source, image_synthesized
