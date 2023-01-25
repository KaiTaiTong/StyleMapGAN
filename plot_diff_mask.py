import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

threshold = 1 / 7

for img_index in tqdm(range(500)):
    try:
        # We want to use reconstructed instead of original so that the base is GAN-generated images
        synthesized_img = plt.imread(
            f"./expr/local_editing/celeba_hq/nose/synthesized_image/{img_index}.png"
        )
        # source_img = plt.imread(f"./expr/local_editing/celeba_hq/nose/source_image/{img_index}.png")
        source_img = plt.imread(
            f"./expr/local_editing/celeba_hq/nose/source_reconstruction/{img_index}.png"
        )

        # Labels of detection takes a binary mask to classify whether this pixel is manipulated or not
        # strict-binary
        # binary_mask = source_img == synthesized_img
        # plt.imshow(~np.all(binary_mask, axis=2), cmap='gray')

        # weak-binary
        binary_mask = np.abs(source_img - synthesized_img)
        binary_mask[np.where(
            binary_mask > np.max(binary_mask) * threshold)] = 1
        binary_mask[np.where(binary_mask != 1)] = 0
        plt.imshow(np.all(binary_mask, axis=2), cmap='gray')
        plt.axis('off')

        plt.savefig(
            f"./expr/local_editing/celeba_hq/nose/binary_mask_1_7/{img_index}.png"
        )

    except FileNotFoundError:
        continue

# python preprocessor/prepare_data.py --out data/celeba_hq/LMDB_train --size "256,1024" data/celeba_hq/raw_images/train