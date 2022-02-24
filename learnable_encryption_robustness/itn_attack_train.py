import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from attack_dataset import get_dataset


def one_epoch(model, data_loader, criterion, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    losses = 0
    data_num = 0
    iter_num = 0

    for images, targets in tqdm(data_loader):
        images, targets = images.to(device), targets.to(device)
        data_num += len(targets)
        iter_num += 1

        if optimizer:
            logits = model(images)
            logits = torch.sigmoid(logits)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                logits = torch.sigmoid(logits)
                loss = criterion(logits, targets)

        losses += loss.item()

    return losses / iter_num


def worker_init_fn(worker_id):
    random.seed(worker_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", default="unet")
    parser.add_argument("--encryption")
    args = parser.parse_args()

    class InverseTransformationNetwork(nn.Module):
        def __init__(self):
            super(InverseTransformationNetwork, self).__init__()
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

    # Constants
    DATA_PATH = "../data/"
    os.environ["TORCH_HOME"] = DATA_PATH
    TRAIN_BATCH_SIZE = 128
    VALID_BATCH_SIZE = 256
    EPOCH_NUM = 50
    CHECKPOINT_FOLDER = f"../model/itn_attack/{args.encryption}"
    NUM_WORKER = 2
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    save_name = "attack_itn.pth"

    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    # Reproducibility
    torch.manual_seed(100)
    random.seed(200)
    np.random.seed(300)
    cudnn.deterministic = True
    cudnn.benchmark = False

    train_ds = get_dataset(
        args.encryption, shuffle=False, data_type="train", key_condition="diff_key"
    )
    valid_ds = get_dataset(
        args.encryption, shuffle=False, data_type="train", key_condition="diff_key"
    )

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_ds)), test_size=0.1, shuffle=True
    )
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKER,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        sampler=train_sampler,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=VALID_BATCH_SIZE,
        num_workers=NUM_WORKER,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        sampler=valid_sampler,
    )

    if args.model == "unet":
        model = InverseTransformationNetwork().to(device)
    else:
        raise ValueError(f"model argment is invalid. {args.model}")

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005
    )
    optim_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 120, 160], gamma=0.2
    )

    valid_loss_list = []
    train_loss_list = []
    valid_accuracy_list = []
    train_accuracy_list = []
    best_valid_loss = None

    for epoch in range(EPOCH_NUM):
        print(f"EPOCH: {epoch}")
        # train
        loss = one_epoch(model, train_dl, criterion, optimizer=optimizer)
        train_loss_list.append(loss)
        print(f"train loss: {loss:.3}")

        # valid
        loss = one_epoch(model, valid_dl, criterion)
        valid_loss_list.append(loss)
        print(f"valid loss: {loss:.3}")

        # step scheduler
        optim_scheduler.step()

        if epoch == 0 or best_valid_loss >= loss:
            print(f"------------------> Update model! {save_name} <------------------")
            best_valid_loss = loss
            torch.save(model.state_dict(), CHECKPOINT_FOLDER + save_name)
