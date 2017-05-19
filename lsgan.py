from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
from dataset import DatasetFromPandas
import models.dcgan as dcgan
import models.mlp as mlp

import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--pos_data', default='/storage03/user_data/liuchen01/creds/train_neg.dat', help='path to dataset')
parser.add_argument('--neg_data', default='/storage03/user_data/liuchen01/creds/train_pos.dat', help='path to dataset')
parser.add_argument('--test_data', default='/storage03/user_data/liuchen01/creds/test_feature.dat', help='path to dataset')
parser.add_argument('--test_label', default='/storage03/user_data/liuchen01/creds/test_labels.dat', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--nSize', type=int, default=148, help='noise size')
parser.add_argument('--nz', type=int, default=148, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true',default=True, help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netP', default='samples/netD_epoch_24.pth', help="path to netP (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true',default=True, help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', default=True,help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt, unknown = parser.parse_known_args()
print(opt)


if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


pos_data = DatasetFromPandas(opt.pos_data)

neg_data = DatasetFromPandas(opt.neg_data)

dataloader = torch.utils.data.DataLoader(pos_data, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
neg_dataloader = torch.utils.data.DataLoader(neg_data, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
test = DatasetFromPandas(opt.test_data)
labels = list(pd.read_csv(opt.test_label,header=None)[0])
testdataloader = torch.utils.data.DataLoader(test, batch_size=len(test),
                                         shuffle=False, num_workers=int(opt.workers))
testdataiter = iter(testdataloader)
testv = Variable(testdataiter.next()).cuda()


ngpu = int(opt.ngpu)
nSize = int(opt.nz)

nz = int(opt.nz)
nSize = int (opt.nSize)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
n_extra_layers = int(opt.n_extra_layers)

netG = nn.Sequential(
    # Z goes into a linear of size: ngf
    nn.Linear(nz, ngf),
    nn.ReLU(True),
    nn.Linear(ngf, ngf),
    nn.ReLU(True),
    nn.Linear(ngf, ngf),
    nn.ReLU(True),
    nn.Linear(ngf, nSize),
    nn.Sigmoid()

)

netD = nn.Sequential(
        nn.Linear(nSize, ndf),
        nn.ReLU(True),
        nn.Linear(ndf, ndf),
        nn.ReLU(True),
        nn.Linear(ndf, ndf),
        nn.ReLU(True),
        nn.Linear(ndf, 1)
)

print (netG)
print (netD)
input = torch.FloatTensor(opt.batchSize, opt.nSize)
noise = torch.FloatTensor(opt.batchSize, nz)
# fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    noise = noise.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)
input = torch.FloatTensor(opt.batchSize, opt.nSize)


if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    # noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

test = DatasetFromPandas(opt.test_data)
labels = list(pd.read_csv(opt.test_label,header=None)[0])
def reset_grad():
    netG.zero_grad()
    netD.zero_grad()
for epoch in range(20):
    data_iter = iter(dataloader)
    neg_iter = iter(neg_dataloader)
    d_step = 5
    i = 0
    while i< len(dataloader):
        j = 0

        while j<d_step and i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            j += 1
            i += 1
            data = data_iter.next()

            # sample data with real and fake
            real_cpu = data

            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)
            try:
                noise = neg_iter.next()
    #                 print (noise.size())
            except:
                neg_iter = iter(neg_dataloader)
                noise = neg_iter.next()
            if opt.cuda:
                noise = noise.cuda()
            # noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise, volatile = True) # totally freeze netG
            fake = Variable(netG(noisev).data)

            # Discriminator
            D_real = netD(inputv)
            D_fake = netD(fake)

            errD = 0.5 * (torch.mean((D_real - 1)**2) + torch.mean(D_fake**2))
            errD.backward()
            optimizerD.step()
            reset_grad()

            ############################
            # (2) Update G network
            ###########################
        try:
                noise = neg_iter.next()

        except:
                neg_iter = iter(neg_dataloader)
                noise = neg_iter.next()
        if opt.cuda:
            noise = noise.cuda()
        noisev = Variable(noise)
        fake = netG(noisev)
        G_fake = netD(fake)
        errG = 0.5 * torch.mean((G_fake - 1)**2)

        errG.backward()
        optimizerG.step()
        reset_grad()
        print('[%d/%d][%d/%d] Loss_D: %f Loss_G: %f '
            % (epoch, opt.niter, i, len(dataloader),
            errD.data[0], errG.data[0]))
        pred_probs = (D_real.cpu().data.numpy())
        print (max(pred_probs),min(pred_probs))
        pred_probs = (G_fake.cpu().data.numpy())
        print (max(pred_probs),min(pred_probs))

        pred_probs = (netD(testv).cpu().data.numpy())
        print (max(pred_probs),min(pred_probs),len(pred_probs))
        pred_probs = (netD(netG(testv)).cpu().data.numpy())
        print (max(pred_probs),min(pred_probs),len(pred_probs))
    if epoch%5==0:
        torch.save(netD.state_dict(), '{0}/lsgan_netD_epoch_{1}.pth'.format(opt.experiment, epoch))

#         pred_probs = (netD(testv.cuda()).cpu().data.numpy())
#         pred_probs = (pred_probs-min(pred_probs))/(max(pred_probs)-min(pred_probs))
#         for i in range(0,10,2):
#             pred = [1 if j>i/10.0 else 0 for j in pred_probs ]
#             print (confusion_matrix(labels,pred))
#             print ("Accuracy, ",  metrics.accuracy_score(labels,pred))
