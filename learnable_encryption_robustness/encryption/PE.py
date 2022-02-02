import numpy as np
import torch


class Warit:
    def __init__(self):
        pass

    @staticmethod
    def negative(x, key):
        key_map = np.random.RandomState(key).randint(0, 2, size=x.shape, dtype=np.int8)
        x_inv = 1 - x
        return np.where(key_map == 1, x_inv, x)

    def encrypt_np(self, x, key):
        return self.negative(x, key)


def batch_encrypt_np(data_batch, key):
    encrypted = []
    enc = Warit()
    for data in data_batch:
        enc_img = enc.encrypt_np(data, key)
        encrypted.append(enc_img.unsqueeze(0))
    return torch.cat(encrypted, dim=0)


def batch_encrypt_np_diff_key(data_batch, key):
    encrypted = []
    for data in data_batch:
        enc = Warit()
        enc_img = enc.encrypt_np(data, key)
        key = key + 1
        encrypted.append(enc_img.unsqueeze(0))
    return torch.cat(encrypted, dim=0)