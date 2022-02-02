from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def load_data(path):
    path = Path(path)
    to_tensor = transforms.ToTensor()
    images = []
    for path_img in path.iterdir():
        img = Image.open(path_img)
        img = to_tensor(img)
        images.append(img)
    return torch.stack(images)


def get_dataset(encrypt, shuffle=False, data_type="train", key_condition="same_key"):
    encrypt_path = f"../encrypted_images/{encrypt}/{data_type}/{key_condition}/"
    plain_path = f"../encrypted_images/plain/{data_type}/"

    protect_ds = load_data(encrypt_path)
    plain_ds = load_data(plain_path)

    protect_ds = (protect_ds - 0.5) / 0.5
    plain_ds = (plain_ds - 0.5) / 0.5

    if shuffle:
        protect_ds = protect_ds[torch.randperm(len(protect_ds))]

    dataset = torch.utils.data.TensorDataset(protect_ds, plain_ds)
    return dataset
