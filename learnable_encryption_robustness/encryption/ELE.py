# https://github.com/MADONOKOUKI/Block-wise-Scrambled-Image-Recognition/blob/0b4606a244e19ca79cc518f9298662c3a4973aea/Blockwise_scramble_LE.py#L1
import numpy as np

from util import BlockScramble


def blockwise_scramble_ele(imgs, block_size=[4, 4, 3], seed=0):
    generator = np.random.default_rng(seed)
    scramble_seeds = generator.integers(100000, size=64)
    x_stack = None
    for k in range(8):
        tmp = None
        # x_stack = None
        for j in range(8):
            bs = BlockScramble(block_size, scramble_seeds[k * 8 + j])
            out = np.transpose(imgs, (0, 2, 3, 1))
            out = out[:, k * 4 : (k + 1) * 4, j * 4 : (j + 1) * 4, :]
            out = bs.Scramble(out.reshape([out.shape[0], 4, 4, 3])).reshape(
                [out.shape[0], 4, 4, 3]
            )
            if tmp is None:
                tmp = out
            else:
                tmp = np.concatenate((tmp, out), axis=2)
        if x_stack is None:
            x_stack = tmp
        else:
            x_stack = np.concatenate((x_stack, tmp), axis=1)
    return x_stack


def block_location_shuffle(img, seed=0):
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(64)

    tmp = img.copy()
    for i in range(64):
        l = permutation[i] // 8
        r = permutation[i] % 8
        a = i // 8
        b = i % 8
        (
            img[:, :, a * 4 : (a + 1) * 4, b * 4 : (b + 1) * 4],
            tmp[:, :, l * 4 : (l + 1) * 4, r * 4 : (r + 1) * 4],
        ) = (
            tmp[:, :, l * 4 : (l + 1) * 4, r * 4 : (r + 1) * 4],
            img[:, :, a * 4 : (a + 1) * 4, b * 4 : (b + 1) * 4],
        )
    return img