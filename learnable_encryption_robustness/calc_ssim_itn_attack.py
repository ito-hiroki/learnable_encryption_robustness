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

from attack_dataset import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--encryption")
args = parser.parse_args()
print(args)


device = "cuda:0"
checkpoint = f"../model/itn_attack/{args.encryption}attack_itn.pth"


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        if args.encryption == "pe":
            self.main = nn.Sequential(
                # for PE
                nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0),
                nn.Tanh(),
            )
        elif args.encryption in ["ele", "etc", "le"]:
            self.main = nn.Sequential(
                # for LE, ELE, (EtC)
                nn.Conv2d(3, 48, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(48),
                nn.ReLU(True),
                nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(48),
                nn.ReLU(True),
                nn.Conv2d(48, 512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.PixelShuffle(4),
                nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"{args.encryption} is invalid.")

    def forward(self, input):
        output = self.main(input)
        return output


model = UNet()
model.load_state_dict(torch.load(checkpoint, map_location=device))
model = model.to(device)


test_ds = get_dataset(
    args.encryption, shuffle=False, data_type="test", key_condition="same_key"
)
test_dl = torch.utils.data.DataLoader(
    test_ds, batch_size=128, shuffle=False, num_workers=2
)

to_pil = ToPILImage()

ssim_list = []
for i, (enc_imgs, plain_imgs) in enumerate(tqdm(test_dl)):
    enc_imgs = enc_imgs.to(device)

    attacked_imgs = model(enc_imgs)

    attacked_imgs = attacked_imgs * 0.5 + 0.5
    plain_imgs = plain_imgs * 0.5 + 0.5

    for idx in range(len(attacked_imgs)):
        attacked_img = np.asarray(to_pil(attacked_imgs[idx]))
        plain_img = np.asarray(to_pil(plain_imgs[idx]))
        ssim_list.append(ssim(attacked_img, plain_img, multichannel=True))

ssim_df = pd.DataFrame(ssim_list, columns=["ssim"])
print(ssim_df.shape)
ssim_df.to_csv(
    f"../ssim_value/ssim_itn_attack_{args.encryption}.csv", index=False, header=True
)
