from __future__ import print_function

import argparse
from operator import index

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from PIL import Image

from attack_dataset import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--encryption")
args = parser.parse_args()
print(args)


ssim_list = []
for idx in tqdm(range(10000)):
    attacked_img = np.asarray(
        Image.open(
            f"../encrypted_images/{args.encryption}/test/same_key/{idx:05}.png"
        )
    )
    plain_img = np.asarray(Image.open(f"../encrypted_images/plain/test/{idx:05}.png"))
    ssim_list.append(ssim(attacked_img, plain_img, multichannel=True))

ssim_df = pd.DataFrame(ssim_list, columns=["ssim"])
print(ssim_df.shape)
ssim_df.to_csv(
    f"../ssim_value/ssim_{args.encryption}.csv", index=False, header=True
)
