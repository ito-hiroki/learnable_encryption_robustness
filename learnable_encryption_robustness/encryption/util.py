# https://github.com/MADONOKOUKI/Block-wise-Scrambled-Image-Recognition/blob/0b4606a244e19ca79cc518f9298662c3a4973aea/learnable_encryption.py#L1
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import tensorflow as tf
import numpy as np
import math


class BlockScramble:
    def __init__(self, block_size, seed):
        self.block_size = block_size
        key = self.genKey(seed)
        self.setKey(key)

    def setKey(self, key):
        self.key = key
        self.rev = key > key.size / 2
        self.invKey = np.argsort(key)

    def genKey(self, seed):
        key = self.block_size[0] * self.block_size[1] * self.block_size[2]
        rng = np.random.default_rng(seed)
        key = rng.permutation(key * 2)
        return key

    def padding(self, X):  # X is [datanum, width, height, channel]
        s = X.shape

        t = s[1] / self.block_size[0]
        d = t - math.floor(t)
        if d > 0:
            paddingSize = self.block_size[0] * (math.floor(t) + 1) - s[1]
            padding = X[:, -1:, :, :]
            padding = np.tile(padding, (1, paddingSize, 1, 1))
            X = np.concatenate((X, padding), axis=1)

        t = s[2] / self.block_size[1]
        d = t - math.floor(t)
        if d > 0:
            paddingSize = self.block_size[1] * (math.floor(t) + 1) - s[2]
            padding = X[:, :, -1:, :]
            padding = np.tile(padding, (1, 1, paddingSize, 1))
            X = np.concatenate((X, padding), axis=2)

        return X

    def Scramble(self, X):
        XX = (X * 255).astype(np.uint8)
        XX = self.doScramble(XX, self.key, self.rev)
        return XX.astype("float32") / 255.0

    def Decramble(self, X):
        XX = (X * 255).astype(np.uint8)
        XX = self.doScramble(XX, self.invKey, self.rev)
        return XX.astype("float32") / 255.0

    def doScramble(self, X, ord, rev):  # X should be uint8
        s = X.shape
        # print(s)
        # print(self.block_size)
        assert X.dtype == np.uint8
        assert s[1] % self.block_size[0] == 0
        assert s[2] % self.block_size[1] == 0
        assert s[3] == self.block_size[2]
        numBlock = np.int32([s[1] / self.block_size[0], s[2] / self.block_size[1]])
        numCh = self.block_size[2]

        X = np.reshape(
            X,
            (
                s[0],
                numBlock[0],
                self.block_size[0],
                numBlock[1],
                self.block_size[1],
                numCh,
            ),
        )
        X = np.transpose(X, (0, 1, 3, 2, 4, 5))
        X = np.reshape(
            X,
            (
                s[0],
                numBlock[0],
                numBlock[1],
                self.block_size[0] * self.block_size[1] * numCh,
            ),
        )
        d = self.block_size[0] * self.block_size[1] * numCh
        # print(X)
        # print(0xF)
        X0 = X & 0xF  # あまりが入る（/16)
        # print(X0)
        X1 = X >> 4  # 16で割ったときの商がはいる
        #  print(X1)
        X = np.concatenate((X0, X1), axis=3)

        X[:, :, :, rev] = (15 - X[:, :, :, rev].astype(np.int32)).astype(np.uint8)
        #   print(ord)
        X = X[:, :, :, ord]
        X[:, :, :, rev] = (15 - X[:, :, :, rev].astype(np.int32)).astype(np.uint8)

        X0 = X[:, :, :, :d]
        X1 = X[:, :, :, d:]
        X = (X1 << 4) + X0

        X = np.reshape(
            X,
            (
                s[0],
                numBlock[0],
                numBlock[1],
                self.block_size[0],
                self.block_size[1],
                numCh,
            ),
        )
        X = np.transpose(X, (0, 1, 3, 2, 4, 5))
        X = np.reshape(
            X,
            (
                s[0],
                numBlock[0] * self.block_size[0],
                numBlock[1] * self.block_size[1],
                numCh,
            ),
        )

        return X