import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils, transforms
import os
import pickle
import random
from training.model import Generator, Encoder
from tqdm import tqdm
import numpy as np
from training.dataset import MultiResolutionDataset, GTMaskDataset
import matplotlib.pyplot as plt
import warnings

MIXING_TYPE = 'local_editing'
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
DATASET_LIST = ['celeba_hq', 'afhq']
THRESHOLD = 1 / 7  # threshold for weak-binary mask


def get_src_ref_pair(total_num_samples, num_pairs):
    """
    Randomly generate num_pairs src and ref indices pair given total number of samples. No 
    duplicated indices could be both src and ref at the same time, and no duplicated pairs 
    """
    indices = set(range(total_num_samples))
    pairing_table = {k: indices-set([k]) for k in indices} # initialize avaliable pairs

    indices1, indices2 = [], []
    for i in range(num_pairs):
        
        # src indices must be selected from available keys
        avaliable_src = [k for k in indices if len(pairing_table[k]) > 0]
        
        if len(avaliable_src) > 0:
            current_src = random.sample(set(avaliable_src), 1)[0]
            current_ref = random.sample(pairing_table[current_src], 1)[0]
            
            indices1.append(current_src)
            indices2.append(current_ref)
            pairing_table[current_src] -= {current_ref}  # update table

        else:
            warnings.warn("Not enough indices to make unique pairs. Total pairs generated: {}".format(len(indices1)))
            break

    return indices1, indices2


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


class Model(nn.Module):

    def __init__(self, device="cuda"):
        super(Model, self).__init__()
        self.g_ema = Generator(
            train_args.size,
            train_args.mapping_layer_num,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            lr_mul=train_args.lr_mul,
            channel_multiplier=train_args.channel_multiplier,
            normalize_mode=train_args.normalize_mode,
            small_generator=train_args.small_generator,
        )
        self.e_ema = Encoder(
            train_args.size,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            channel_multiplier=train_args.channel_multiplier,
        )

    def forward(self, input, mode):

        if mode == "projection":
            fake_stylecode = self.e_ema(input)
            return fake_stylecode

        elif mode == "local_editing":
            w1, w2, mask = input
            w1, w2, mask = w1.unsqueeze(0), w2.unsqueeze(0), mask.unsqueeze(0)

            if train_args.dataset == "celeba_hq":
                mixed_image = self.g_ema(
                    [w1, w2],
                    input_is_stylecode=True,
                    mix_space="w_plus",
                    mask=mask,
                )[0]

            elif train_args.dataset == "afhq":
                mixed_image = self.g_ema([w1, w2],
                                         input_is_stylecode=True,
                                         mix_space="w",
                                         mask=mask)[0]

            recon_img_src, _ = self.g_ema(w1, input_is_stylecode=True)
            recon_img_ref, _ = self.g_ema(w2, input_is_stylecode=True)

            return mixed_image, recon_img_src, recon_img_ref


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
])

if __name__ == "__main__":

    ckpt_path = "./expr/checkpoints/celeba_hq_256_8x8.pt"
    lmdb_file = "data/celeba_hq/LMDB_val"  # input images in mdb file
    num_samples_per_class = 100
    batch = 16
    num_workers = 10
    save_image_dir = "expr"
    # local_editing_part = "nose"
    # local_editing_part = "ear"
    local_editing_part = "eye"
    # local_editing_part = "lip"

    device = "cuda"

    # Load model parameters
    ckpt = torch.load(ckpt_path)
    train_args = ckpt["train_args"]
    model = Model().to(device)
    model.g_ema.load_state_dict(ckpt["g_ema"])
    model.e_ema.load_state_dict(ckpt["e_ema"])
    model.eval()

    # Set output path
    save_image_dir = os.path.join(save_image_dir, MIXING_TYPE,
                                  train_args.dataset)
    if train_args.dataset == "afhq":
        save_image_dir = os.path.join(save_image_dir)
        for kind in [
                "output_diff_mask",  # diff between reconstructed src. and output binary mask
                "editing_mask",  # binary mask for local editing region
                "reference_reconstruction",  # reconstructed ref. img
                "source_reconstruction",  # reconstructed src. img         
                "synthesized_image"  # local edited img
        ]:
            os.makedirs(os.path.join(save_image_dir, kind), exist_ok=True)
    else:  # celeba_hq
        save_image_dir = os.path.join(save_image_dir, local_editing_part)
        for kind in [
                "output_diff_mask",  # diff between reconstructed src. and output binary mask
                "editing_mask",  # binary mask for local editing region
                "reference_reconstruction",  # reconstructed ref. img
                "source_reconstruction",  # reconstructed src. img         
                "synthesized_image"  # local edited img
        ]:
            os.makedirs(os.path.join(save_image_dir, kind), exist_ok=True)
        mask_path_base = f"data/{train_args.dataset}/local_editing"

    # Prepare Dataset
    if train_args.dataset == "celeba_hq":
        assert "celeba_hq" in lmdb_file

        # CelebA dataset contains an RGB img, and a classification mask pair
        # parts_index represents the indices for each parts in the mask
        dataset = GTMaskDataset(lmdb_file, transform, train_args.size)

        parts_index = {
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
        }

    # afhq, coarse(half-and-half) masks
    else:
        assert "afhq" in lmdb_file and "afhq" == train_args.dataset
        dataset = MultiResolutionDataset(lmdb_file, transform, train_args.size)

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

    total_latents = torch.Tensor().to(device)  # Outputs of models
    real_imgs = torch.Tensor().to(device)  # real images

    if train_args.dataset == "afhq":
        masks = (-2 * torch.ones(n_sample, train_args.size,
                                 train_args.size).to(device).float())
        mix_type = list(range(n_sample))
        random.shuffle(mix_type)
        horizontal_mix = mix_type[:n_sample // 2]
        vertical_mix = mix_type[n_sample // 2:]

        masks[horizontal_mix, :, train_args.size // 2:] = 2
        masks[vertical_mix, train_args.size // 2:, :] = 2
    else:
        masks = torch.Tensor().to(device).long()

    # Local editing
    with torch.no_grad():

        # Concatenate real imgs and its latents
        for i, real_img in enumerate(tqdm(loader, mininterval=1)):
            if train_args.dataset == "celeba_hq":
                real_img, mask = real_img  # unpack image and mask
                mask = mask.to(device)
                masks = torch.cat([masks, mask], dim=0)
            real_img = real_img.to(device)

            latents = model(real_img, "projection")

            total_latents = torch.cat([total_latents, latents], dim=0)
            real_imgs = torch.cat([real_imgs, real_img], dim=0)

        # We don't want to hand pick the pairing based on similarity, because
        # we want the prediction model to be trained on a mixture of good and
        # bad samples

        # Randomly pick unique pairs of src and ref
        indices1, indices2 = get_src_ref_pair(len(total_latents), num_samples_per_class)

        # if train_args.dataset == "afhq":
        #     # change it later
        #     indices = list(range(len(total_latents)))
        #     random.shuffle(indices)
        #     indices1 = indices[:len(total_latents) // 2] # first half
        #     indices2 = indices[len(total_latents) // 2:] # second half

        # else:
        # with open(
        #         f"{mask_path_base}/celeba_hq_test_GT_sorted_pair.pkl",
        #         "rb",
        # ) as f:
        #     sorted_similarity = pickle.load(f)

        # indices1 = []
        # indices2 = []
        # for (i1, i2), _ in sorted_similarity[local_editing_part]:
        #     indices1.append(i1)
        #     indices2.append(i2)

        for loop_i, (index1, index2) in tqdm(enumerate(zip(indices1,
                                                           indices2)),
                                             total=n_sample):
            src_img = real_imgs[index1]
            ref_img = real_imgs[index2]

            if train_args.dataset == "celeba_hq":
                mask1_logit = masks[index1]
                mask2_logit = masks[index2]

                mask1 = -torch.ones(mask1_logit.shape).to(
                    device)  # initialize with -1
                mask2 = -torch.ones(mask2_logit.shape).to(
                    device)  # initialize with -1

                for label_i in parts_index[local_editing_part]:
                    mask1[(mask1_logit == label_i) == True] = 1
                    mask2[(mask2_logit == label_i) == True] = 1

                mask = mask1 + mask2
                mask = mask.float()
            elif train_args.dataset == "afhq":
                mask = masks[index1]

            mixed_image, recon_img_src, recon_img_ref = model(
                (total_latents[index1], total_latents[index2], mask),
                "local_editing",
            )

            save_images(
                [
                    mixed_image[0],
                    recon_img_src[0],
                    recon_img_ref[0],
                ],
                [
                    f"{save_image_dir}/synthesized_image/{index1}.png",
                    f"{save_image_dir}/source_reconstruction/{index1}.png",
                    f"{save_image_dir}/reference_reconstruction/{index1}.png",
                ],
            )

            mask[mask < -1] = -1
            mask[mask > -1] = 1

            # Generate output difference binary mask (weak)
            output_binary_mask = torch.abs(mixed_image[0] - recon_img_src[0])
            output_binary_mask[torch.where(
                output_binary_mask > torch.max(output_binary_mask) *
                THRESHOLD)] = 1
            output_binary_mask[torch.where(output_binary_mask != 1)] = 0
            output_binary_mask = torch.all(output_binary_mask.type(
                torch.uint8),
                                           axis=0).float()

            output_binary_mask[output_binary_mask < 1] = -1
            output_binary_mask[output_binary_mask > 0] = 1

            # #################### View #################### 
            # import matplotlib.pyplot as plt
            # plt.imshow(output_binary_mask.cpu(), cmap='gray')

            # import matplotlib.pyplot as plt
            # plt.imshow(np.rot90(np.swapaxes(real_img[2, :].cpu(), 0, -1), 3))
            # plt.imshow(mask[2, :].cpu(), cmap='gray')

            # for i, real_img in enumerate(dataset):
            #     real_img, mask = real_img
            #     fig, (ax1, ax2) = plt.subplots(1, 2)
            #     ax1.imshow(np.rot90(np.swapaxes(real_img.cpu(), 0, -1), 3))
            #     ax2.imshow(mask.cpu(), cmap='gray')
            #     plt.show()

            save_images([output_binary_mask, mask], [
                f"{save_image_dir}/output_diff_mask/{index1}.png",
                f"{save_image_dir}/editing_mask/{index1}.png",
            ])
