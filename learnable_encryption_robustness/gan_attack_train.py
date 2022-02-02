# Waritさんのスクリプトに手を色々と加えた
from __future__ import print_function

import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from attack_dataset import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--encryption")
parser.add_argument(
    "--workers", type=int, help="number of data loading workers", default=2
)
parser.add_argument("--batchSize", type=int, default=64, help="input batch size")
parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector")
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument(
    "--niter", type=int, default=25, help="number of epochs to train for"
)
parser.add_argument(
    "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
)
parser.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
)
parser.add_argument("--cuda", action="store_true", help="enables cuda")
parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
parser.add_argument("--netG", default="", help="path to netG (to continue training)")
parser.add_argument("--netD", default="", help="path to netD (to continue training)")
parser.add_argument(
    "--outf",
    default="../model/gan_attack/",
    help="folder to output images and model checkpoints",
)
parser.add_argument("--manualSeed", type=int, help="manual seed")


opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf + opt.encryption)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


train_ds = get_dataset(
    opt.encryption, shuffle=True, data_type="train", key_condition="same_key"
)
train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers)
)
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        if opt.encryption == "pe":
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
        elif opt.encryption in ["ele", "etc", "le"]:
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
            raise ValueError(f"{opt.encryption} is invalid.")

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != "":
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ndf , 1, 4, 1, 0, bias=False),
            # state size. (ndf*4) x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),  # ↑のspectral_normに安定性のため変更
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
            nn.MaxPool2d(3),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class ProbLoss(nn.Module):  # ref: https://qiita.com/koshian2/items/9add09dabb44cab2b4c0
    def __init__(self, opt):
        assert opt["loss_type"] in ["bce", "hinge"]
        super().__init__()
        self.loss_type = opt["loss_type"]
        self.device = opt["device"]
        self.ones = torch.ones(opt["batch_size"]).to(opt["device"])
        self.zeros = torch.zeros(opt["batch_size"]).to(opt["device"])
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, logits, condition):
        assert condition in ["gen", "dis_real", "dis_fake"]
        batch_len = len(logits)

        if self.loss_type == "bce":
            if condition in ["gen", "dis_real"]:
                return self.bce(logits, self.ones[:batch_len])
            else:
                return self.bce(logits, self.zeros[:batch_len])

        elif self.loss_type == "hinge":
            # SPADEでのHinge lossを参考に実装
            # https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py
            if condition == "gen":
                # Generatorでは、本物になるようにHinge lossを返す
                return -torch.mean(logits)
            elif condition == "dis_real":
                minval = torch.min(logits - 1, self.zeros[:batch_len])
                return -torch.mean(minval)
            else:
                minval = torch.min(-logits - 1, self.zeros[:batch_len])
                return -torch.mean(minval)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != "":
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

# fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0
# criterion_class = nn.CrossEntropyLoss()
# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


for epoch in range(opt.niter):
    # for i, (enc_img, plain) in enumerate(train_dl):
    for i, (enc_img, plain) in enumerate(train_dl):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = plain.to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device).float()
        output = netD(real_cpu)
        errD_real = criterion(output, label)

        # 追加分
        # errD_real = torch.nn.ReLU()(1.0 - output).mean()

        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        # noise = torch.randn(batch_size, nz, 1, 1, device=device)

        # enc_img = enc_img.view(-1,3072,1,1)
        enc_img = enc_img.to(device)
        #
        fake = netG(enc_img)
        # net.zero_grad()
        # with torch.no_grad():

        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)  # *0.1 + class_loss*0.9

        # 追加分
        # errD_fake = torch.nn.ReLU()(1.0 + output).mean()

        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)  # *0.1 + class_loss*0.9
        # errG = output.mean()  # *0.1 + class_loss*0.9
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print(
            "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
            % (
                epoch,
                opt.niter,
                i,
                len(train_dl),
                errD.item(),
                errG.item(),
                D_x,
                D_G_z1,
                D_G_z2,
            )
        )
        if i % 100 == 0:
            vutils.save_image(
                real_cpu,
                f"{opt.outf}/{opt.encryption}/real_samples.png",
                normalize=True,
            )
            fake = netG(enc_img)
            vutils.save_image(
                fake.detach(),
                f"{opt.outf}/{opt.encryption}/fake_samples_epoch_{epoch}.png",
                normalize=True,
            )

    # do checkpointing
    torch.save(netG.state_dict(), f"{opt.outf}/{opt.encryption}/netG_epoch_{epoch}.pth")
    torch.save(netD.state_dict(), f"{opt.outf}/{opt.encryption}/netD_epoch_{epoch}.pth")
