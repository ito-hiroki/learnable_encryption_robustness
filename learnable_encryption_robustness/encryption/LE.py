# https://github.com/MADONOKOUKI/Block-wise-Scrambled-Image-Recognition/blob/0b4606a244e19ca79cc518f9298662c3a4973aea/Blockwise_scramble_LE.py#L1
import numpy as np

from util import BlockScramble


def blockwise_scramble_le(imgs, block_size=[4, 4, 3], seed=0):
    x_stack = None
    for k in range(8):
        tmp = None
        # x_stack = None
        for j in range(8):
            bs = BlockScramble(block_size, seed)
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
