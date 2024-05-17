from __future__ import print_function
from six.moves import range
import argparse
import torch.backends.cudnn as cudnn
import torch
import wandb
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import time
from PIL import Image, ImageFont, ImageDraw
from copy import deepcopy
from transformers import BertTokenizer, BertModel
from scipy import linalg
from collections import OrderedDict
from miscc.config import cfg
from miscc.utils import mkdir_p

from efficient_net import build_efficientnet_lite, load_checkpoint

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# if cfg.TRAIN.OPT == 'bcr':
#     from model_bcr import G_NET, D_NET64, D_NET128, D_NET256
# else:
from model import G_NET, D_NET64, D_NET128, D_NET256
from text_encoder import TextEncoder
from diffaug import DiffAugment
from metrics import InceptionV3, compute_inception_score, calculate_fretchet


# ############# Shared Function ############
def compute_mean_covariance(img):
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose(1, 2)
    # batch_size * channel_num * channel_num
    covariance = torch.bmm(img_hat, img_hat_transpose)  # 共分散計算
    covariance = covariance / num_pixels

    return mu, covariance


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)  # 直交
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
        # はテンソルを平均 mean、分散 std**2 の正規分布で初期化

        m.bias.data.fill_(0.0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def get_gradient_ratios(lossA, lossB, x_f, eps=1e-6):
    grad_lossA_xf = torch.autograd.grad(torch.sum(lossA), x_f, retain_graph=True)[0]
    grad_lossB_xf = torch.autograd.grad(torch.sum(lossB), x_f, retain_graph=True)[0]
    gamma = grad_lossA_xf / grad_lossB_xf

    return gamma


def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', prediction.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def load_network(gpus, feature_sizes=None):
    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    print(netG)

    netsD = []

    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_NET64())
        # netsD.append(PD_NEThttps://www.notion.so/21726899a1344ef5bbd3bf69a7847918?pvs=464(feature_sizes[3]))
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_NET128())
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_NET256())

    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
    print('# of netsD', len(netsD))

    count = 0
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('load', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1

    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)

    inception_model = InceptionV3()

    if cfg.CUDA:
        netG.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()
        inception_model = inception_model.cuda()
    inception_model.eval()

    return netG, netsD, len(netsD), inception_model, count


def define_optimizers(netG, netsD):
    optimizersD = []
    num_Ds = len(netsD)
    if cfg.TRAIN.TYPE == 'wgan':
        for i in range(num_Ds):
            opt = optim.RMSprop(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             )
            optimizersD.append(opt)
        optimizerG = optim.RMSprop(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            )
    else:
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        # G_opt_paras = []
        # for p in netG.parameters():
        #     if p.requires_grad:
        #         G_opt_paras.append(p)
        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
    return optimizerG, optimizersD


def save_model(netG, avg_param_G, netsD, epoch, model_dir):
    load_params(netG, avg_param_G)
    torch.save(
        netG.state_dict(),
        '%s/netG_%d.pth' % (model_dir, epoch))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(
            netD.state_dict(),
            '%s/netD%d.pth' % (model_dir, i))


def save_img_results(imgs_tcpu, fake_imgs, num_imgs,
                     count, epoch, image_dir):
    num = cfg.TRAIN.VIS_COUNT
    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/real_samples.png' % (image_dir),
        normalize=True)
    real_img_set = vutils.make_grid(real_img).numpy()
    real_img_set = real_img_set * 255
    real_img_set = real_img_set.astype(np.uint8)
    

    for i in range(num_imgs):
        fake_img = fake_imgs[i][0:num]
        # The range of fake_img.data (i.e., self.fake_imgs[i][0:num])
        # is still [-1. 1]...
        if count:
            vutils.save_image(
                fake_img.data, '%s/epoch_%03d_count_%09d_fake_samples%d.png' %
                               (image_dir, epoch, count, i), normalize=True)
        else:
            vutils.save_image(
                fake_img.data, '%s/epoch_%03d_fake_samples%d.png' %
                               (image_dir, epoch, i), normalize=True)
        fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()
        fake_img_set = (fake_img_set + 1) * 255 / 2
        fake_img_set = fake_img_set.astype(np.uint8)


# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, img_size, checkpoint_efficient_net, i2w):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)


        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.img_size = img_size

    def build_models(self):
        netG = G_NET()
        netG.apply(weights_init)
        netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
        nerG = netG.module
        # print(netG)

        netsD = []

        if cfg.TREE.BRANCH_NUM > 0:
            netsD.append(D_NET64())
            # netsD.append(PD_NET64(feature_sizes[3]))
        if cfg.TREE.BRANCH_NUM > 1:
            netsD.append(D_NET128())
        if cfg.TREE.BRANCH_NUM > 2:
            netsD.append(D_NET256())

        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=self.gpus)
            netsD[i] =netsD[i].module
        # print('# of netsD', len(netsD))

        count = 0
        epoch = 0
        if cfg.TRAIN.PATH != '':
            if cfg.TRAIN.NET_G != '':
                G_path = os.path.join(cfg.TRAIN.PATH, cfg.TRAIN.NET_G)
                state_dict = torch.load(G_path)
                netG.load_state_dict(state_dict)
                print('load', cfg.TRAIN.NET_G)

                istart = cfg.TRAIN.NET_G.rfind('_') + 1
                iend = cfg.TRAIN.NET_G.rfind('.')
                epoch = cfg.TRAIN.NET_G[istart:iend]
                epoch = int(epoch) 

            if cfg.TRAIN.NET_D != '':
                for i in range(len(netsD)):
                    print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
                    netD = '%s%d.pth' % (cfg.TRAIN.NET_D, i)
                    D_path = os.path.join(cfg.TRAIN.PATH, netD)
                    state_dict = torch.load(D_path)                  
                    netsD[i].load_state_dict(state_dict)

        # block_idx = InceptionFID.BLOCK_INDEX_BY_DIM[2048]
#         fid_model = InceptionFID([3]) # 2048
#         fid_model = fid_model.cuda()
        
        


        text_encoder = TextEncoder(
            data_dir='word2vec',
            emb_dim=300,
            hid_dim=100,
            z_dim=256,  # 1024
            # word2vec_file=ckpt_args.word2vec_file,
            with_attention=True,
            ingr_enc_type='fc'
        )
        inception_model = InceptionV3()
        if cfg.CUDA:
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
            inception_model = inception_model.cuda()
          
      
        inception_model.eval()


        return netG, netsD, len(netsD), text_encoder, inception_model, epoch, count
    
    

    def prepare_data(self, data, device='cuda'):
        title, ing, n_ing, ins_feat, imgs, w_imgs, _ = data
        ins_feat = torch.squeeze(ins_feat, dim=1)
        #print(ins[:1])
        # txt = torch.cat((ins, ing), 1)  # 16, 1, 512
        real_vimgs, wrong_vimgs = [], []
        # imgs = torch.stack(imgs, dim=0)
        # w_imgs = torch.stack(w_imgs, dim=0)
        # imgs = imgs[0]
        # w_imgs = w_imgs[0]
#         print('len of imgs: ', len(imgs))
#         b = np.array(imgs)
#         print('np', imgs.size)
        for i in range(len(imgs)):
            real_vimgs.append(imgs[i].to(device))
            wrong_vimgs.append(w_imgs[i].to(device))
        # print(real_vimgs[:2])
        vins_feat = ins_feat.to(device)
        vtitle = title.clone().detach().to(device)
        ving = ing.clone().detach().to(device)
#         print(vtitle.shape)
#         print(ving.shape)
        
        # vtitle = [x.to(device) for x in title]
        # ving = [x.to(device) for x in ing]  
        return vtitle, ving, vins_feat, real_vimgs, wrong_vimgs, imgs

    def train_Dnet(self, idx, count):
        flag = 1  # count % 100
        batch_size = self.real_imgs[0].size(0)  # 1: 24
        criterion, mu, crloss = self.criterion, self.mu, self.crloss  # BCELoss  24, 128

        netD, optD = self.netsD[idx], self.optimizersD[idx]  # Adam
        real_imgs = self.real_imgs[idx]  # 24,3,64,64
        
#         print('real_imgs: ',real_imgs.shape)
        wrong_imgs = self.wrong_imgs[idx]  # 24,3,64,64
       
        fake_imgs = self.fake_imgs[idx]  # 24,3,64,64
        
        # get efficient net features
        netD.zero_grad()
        # Forward
        real_labels = self.real_labels[:batch_size].to('cuda')  # 24
        fake_labels = self.fake_labels[:batch_size].to('cuda')  # 24
        # logits:[[conditional], [unconditional]]
        if cfg.TRAIN.OPT == 'diffaug':
            real_aug = DiffAugment(real_imgs, policy='translation')
            wrong_aug = DiffAugment(wrong_imgs, policy='translation')
            fake_aug = DiffAugment(fake_imgs, policy='translation')
            
            real_logits = netD(real_aug.detach(), mu.detach())  # list:2
            errD_real = criterion(real_logits[0], real_labels)

            wrong_logits = netD(wrong_aug.detach(), mu.detach())  # list:2
            errD_wrong = criterion(wrong_logits[0], fake_labels)

            fake_logits = netD(fake_aug.detach(), mu.detach())  # list:2
            errD_fake = criterion(fake_logits[0], fake_labels)
        else:
            real_logits = netD(real_imgs, mu.detach())  # list:2
            errD_real = criterion(real_logits[0], real_labels)

            wrong_logits = netD(wrong_imgs, mu.detach())  # list:2
            errD_wrong = criterion(wrong_logits[0], fake_labels)

            fake_logits = netD(fake_imgs.detach(), mu.detach())  # list:2
            errD_fake = criterion(fake_logits[0], fake_labels)

        # Consistency regularization for real and fake images (bCR)
        crL = 0
        if cfg.TRAIN.OPT == 'bcr': 
            real_aug = DiffAugment(real_imgs, policy='zt')  # Random translation with mirrored padding
            real_aug_logits = netD(real_aug.detach(), mu.detach())
            # bCR: Calculate L_real: |D(x) − D(T(x))|^2
            crL_real = crloss(real_logits[0].data, real_aug_logits[0])  # MSE平均二乗誤差
            # wrong_aug = DiffAugment(wrong_imgs, policy='translation')
            # wrong_aug_logits = netD(wrong_aug.detach(), mu.detach())
            # crL_wrong = crloss(wrong_logits[1].data, wrong_aug_logits[1])
            
            fake_aug = DiffAugment(fake_imgs, policy='zt')
            fake_aug_logits = netD(fake_aug.detach(), mu.detach())
            
            crL_fake = crloss(fake_logits[0].data, fake_aug_logits[0])
            if count == 1:
                print(crL_real)
                print(crL_fake)
                # print(crL_wrong)
            lambda_cr = cfg.TRAIN.COEFF.CR
            crL = lambda_cr * (crL_real + crL_fake)


        
        if len(real_logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
            errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                               criterion(real_logits[1], real_labels)

            errD_wrong_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                                criterion(wrong_logits[1], real_labels)
            errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                               criterion(fake_logits[1], fake_labels)
            #
            errD_cond = errD_real + errD_wrong + errD_fake
            errD_uncond = errD_real_uncond + errD_fake_uncond + errD_wrong_uncond
            errD_real = errD_real + errD_real_uncond  # float
            errD_wrong = errD_wrong + errD_wrong_uncond
            errD_fake = errD_fake + errD_fake_uncond
            #

            errD = errD_real + errD_wrong + errD_fake
            if cfg.DEBUG == 0:
                wandb.log({
                    f'errD_cond{idx}': errD_cond,
                    f'errD_uncond{idx}': errD_uncond,
                    f'errD{idx}': errD,
                    f'batch_idx': count,  # batch_idx
                })
        else:
            errD = errD_real + 0.5 * (errD_wrong + errD_fake)
            if cfg.DEBUG == 0:
                wandb.log({
                    f'errD{idx}': errD,
                    f'batch_idx': count,  # batch_idx
                })
                if crL != 0:
                    wandb.log({
                        f'crL{idx}': crL,
                        f'batch_idx': count,  # batch_idx
                    })
        errD = errD + crL
        # backward
        errD.backward()
        # update parameters
        optD.step()

        return errD

    def train_Gnet(self, count):
        self.netG.zero_grad()
        errG_total = 0
        flag = 1  # count % 100
        batch_size = self.real_imgs[0].size(0)
        criterion, mu, logvar = self.criterion, self.mu, self.logvar
        real_labels = self.real_labels[:batch_size].to('cuda')
        for i in range(self.num_Ds):
            fake_imgs = self.fake_imgs[i]
            if cfg.TRAIN.OPT == 'diffaug':
                fake_aug = DiffAugment(fake_imgs, policy='translation')
                outputs = self.netsD[i](fake_aug.detach(), mu.detach())
                errG = criterion(outputs[0], real_labels)
#                 crL_fake = crloss(outputs[0], fake_aug_logits[0])
#                 crL = 10 * (crL_real + 0.5*(crL_wrong + crL_fake))# 'color,'
            else:
                outputs = self.netsD[i](fake_imgs, mu.detach())
                errG = criterion(outputs[0], real_labels)
            
            
            if len(outputs) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
                errG_patch = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                             criterion(outputs[1], real_labels)
                errG = errG + errG_patch
            errG_total = errG_total + errG
            if cfg.DEBUG == 0:
                wandb.log({
                    f'errG': errG,
                    f'batch_idx': count,  # batch_idx
                })

        # Compute color consistency losses
        if cfg.TRAIN.COEFF.COLOR_LOSS > 0:
            if self.num_Ds > 1:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-1])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-2].detach())
                like_mu2 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov2 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                            nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu2 + like_cov2
                # if flag == 0:
                #     
            if self.num_Ds > 2:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-2])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-3].detach())
                like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                            nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu1 + like_cov1
                # if flag == 0:


        kl_loss = KL_loss(mu, logvar) * cfg.TRAIN.COEFF.KL
        errG_total = errG_total + kl_loss
        errG_total.backward()
        self.optimizerG.step()
        return kl_loss, errG_total

    def train(self):
        self.netG, self.netsD, self.num_Ds, \
            self.text_encoder, self.inception_model, epoch, count = self.build_models()
        
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.BCELoss()
        self.crloss = nn.MSELoss()
        # self.real_labels = \
        #     Variable(torch.FloatTensor(self.batch_size).fill_(1))
        # self.fake_labels = \
        #     Variable(torch.FloatTensor(self.batch_size).fill_(0))

        # Prepare labels
        if cfg.LABELS == 'original':
            self.real_labels = torch.FloatTensor(self.batch_size).fill_(
                1)  # (torch.FloatTensor(args.batch_size).uniform_() < 0.9).float() #
            self.fake_labels = torch.FloatTensor(self.batch_size).fill_(
                0)  # (torch.FloatTensor(args.batch_size).uniform_() > 0.9).float() #
        elif cfg.LABELS == 'R-smooth':
            self.real_labels = torch.FloatTensor(self.batch_size).fill_(1) - (
                    torch.FloatTensor(self.batch_size).uniform_() * 0.1)
            self.fake_labels = (torch.FloatTensor(self.batch_size).uniform_() * 0.1)


        self.gradient_one = torch.FloatTensor([1.0])
        self.gradient_half = torch.FloatTensor([0.5])

        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = \
            Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))

        if cfg.CUDA:
            self.criterion.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            self.gradient_one = self.gradient_one.cuda()
            self.gradient_half = self.gradient_half.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        preds_is = []
        preds_fid_fake = []
        preds_fid_real = []
        count = epoch * (self.num_batches) + 1
        start_epoch = epoch + 1
        # start_epoch = start_count // (self.num_batches)
        print('Start from count ', count)
        print('Start from epoch', start_epoch)
        for epoch in range(start_epoch, self.max_epoch):
            # print('Epoch ', epoch)
            start_t = time.time()
            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # print(count)
                #######################################################
                # (0) Prepare training data
                ######################################################
                # print("Preparing training data...")
                data = next(data_iter)

                title, ing, ins_feat, self.real_imgs, self.wrong_imgs, self.imgs_tcpu = self.prepare_data(data)
                # self.imgs_tcpu: real images in cpu
                with torch.no_grad():
                    txt_feat = self.text_encoder([title, ing])
#                 print(txt_feat.shape)
#                 print(ins_feat.shape)
                self.txt_embedding = torch.cat([txt_feat, ins_feat], dim=1)
                #######################################################
                # (1) Generate fake images
                ######################################################
                # print("Generating fake imgs...")
                noise.data.normal_(0, 1)
                self.fake_imgs, self.mu, self.logvar = \
                    self.netG(noise, self.txt_embedding)

                #######################################################
                # (2) Update D network
                ######################################################
                # print("Updating networks...")
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i, count)
                    # errD = self.train_Dnet_projected(i, count)
                    errD_total += errD

                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                kl_loss, errG_total = self.train_Gnet(count)
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    # avg_p.mul_(0.999).add_(0.001, p.data)
                    avg_p.mul_(0.999).add_(p.data, alpha=0.001)

                # for inception score
                # pred_is = self.inception_model(self.fake_imgs[-1].detach())
                pred_is, pred_fid_fake = self.inception_model(self.fake_imgs[-1].detach())
#                 if count == 1:
#                     print('pred_fid_fake:', pred_fid_fake.shape)
                _, pred_fid_real = self.inception_model(self.real_imgs[-1].detach())
#                 print('pred_fid_real:', pred_fid_real.shape)
                # print(pred_fid_real)

                preds_is.append(pred_is.data.cpu().numpy())
                #  (BS * N, 2048) 
         
                preds_fid_fake.append(pred_fid_fake.data.cpu().numpy().reshape(pred_fid_fake.size(0), -1))
                preds_fid_real.append(pred_fid_real.data.cpu().numpy().reshape(pred_fid_real.size(0), -1))

                if cfg.DEBUG == 0:
                    wandb.log({
                        f'D_loss': errD_total,
                        f'G_loss': errG_total,
                        f'batch_idx': count,  # batch_idx
                    })
                
                
                step = step + 1
                count = count + 1
            if len(preds_is) > 1000:
                preds_is = np.concatenate(preds_is, 0)
                mean, std = compute_inception_score(preds_is, 10)
                preds_fid_fake = np.concatenate(preds_fid_fake, 0)
                preds_fid_real = np.concatenate(preds_fid_real, 0)  # BS * N, 2048
                fid_score = calculate_fretchet(preds_fid_real, preds_fid_fake)

                #
                mean_nlpp, std_nlpp = \
                    negative_log_posterior_probability(preds_is, 10)
                
                if cfg.DEBUG == 0:
                    wandb.log({
                        f'Inception_mean': mean,
                        f'FID_score': fid_score,
                        f'epoch': epoch,  
                        })
                #
                
                print('''IS: %.2f FID: %.2f '''
                  % (mean.item(), fid_score.item()))
                preds_is = []
                preds_fid_fake = []
                preds_fid_real = []
     
            end_t = time.time()
            if epoch % 10 == 0:
                save_model(self.netG, avg_param_G, self.netsD, epoch, self.model_dir)
                # Save images
                backup_para = copy_G_params(self.netG)
                load_params(self.netG, avg_param_G)
                #
                self.fake_imgs, _, _ = \
                    self.netG(fixed_noise, self.txt_embedding)
                save_img_results(self.imgs_tcpu, self.fake_imgs, self.num_Ds,
                                 count, epoch, self.image_dir)
                #
                load_params(self.netG, backup_para)
            
            print('''[%d/%d][%d]
                         Loss_D: %.2f Loss_G: %.2f Time: %.2fs
                      '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(), end_t - start_t))

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)

    def save_superimages(self, images_list, filenames,
                         save_dir, split_dir, imsize):
        batch_size = images_list[0].size(0)
        num_sentences = len(images_list)
        for i in range(batch_size):
            s_tmp = '%s/super/%s/%s' % \
                    (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            #
            savename = '%s_%d.png' % (s_tmp, imsize)
            super_img = []
            for j in range(num_sentences):
                img = images_list[j][i]
                # print(img.size())
                img = img.view(1, 3, imsize, imsize)
                # print(img.size())
                super_img.append(img)
                # break
            super_img = torch.cat(super_img, 0)
            vutils.save_image(super_img, savename, nrow=10, normalize=True)

    def save_singleimages(self, images, filenames,
                          save_dir, split_dir, sentenceID, imsize):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' % \
                    (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
            # range from [-1, 1] to [0, 255]
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def evaluate(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            if split_dir == 'test':
                split_dir = 'valid'
            netG = G_NET()
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            print(netG)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = \
                torch.load(cfg.TRAIN.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load ', cfg.TRAIN.NET_G)

            # the path to save generated images
            s_tmp = cfg.TRAIN.NET_G
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            iteration = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/iteration%d' % (s_tmp, iteration)

            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(self.batch_size, nz))
            if cfg.CUDA:
                netG.cuda()
                noise = noise.cuda()

            # switch to evaluate mode
            netG.eval()
            for step, data in enumerate(self.data_loader, 0):
                imgs, t_embeddings, filenames = data
                if cfg.CUDA:
                    t_embeddings = Variable(t_embeddings).cuda()
                else:
                    t_embeddings = Variable(t_embeddings)
                # print(t_embeddings[:, 0, :], t_embeddings.size(1))

                embedding_dim = t_embeddings.size(1)
                batch_size = imgs[0].size(0)
                noise.data.resize_(batch_size, nz)
                noise.data.normal_(0, 1)

                fake_img_list = []
                for i in range(embedding_dim):
                    fake_imgs, _, _ = netG(noise, t_embeddings[:, i, :])
                    if cfg.TEST.B_EXAMPLE:
                        # fake_img_list.append(fake_imgs[0].data.cpu())
                        # fake_img_list.append(fake_imgs[1].data.cpu())
                        fake_img_list.append(fake_imgs[2].data.cpu())
                    else:
                        self.save_singleimages(fake_imgs[-1], filenames,
                                               save_dir, split_dir, i, 256)
                        # self.save_singleimages(fake_imgs[-2], filenames,
                        #                        save_dir, split_dir, i, 128)
                        # self.save_singleimages(fake_imgs[-3], filenames,
                        #                        save_dir, split_dir, i, 64)
                    # break
                if cfg.TEST.B_EXAMPLE:
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 64)
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 128)
                    self.save_superimages(fake_img_list, filenames,
                                          save_dir, split_dir, 256)
                    
#
# # ################# WGAN ############################ #
# class condWGANTrainer(object):
#     def __init__(self, output_dir, data_loader, img_size, checkpoint_efficient_net):
#         if cfg.TRAIN.FLAG:
#             self.model_dir = os.path.join(output_dir, 'models/Model')
#             self.image_dir = os.path.join(output_dir, 'Image')
#             self.log_dir = os.path.join(output_dir, 'Log')
#             mkdir_p(self.model_dir)
#             mkdir_p(self.image_dir)
#             mkdir_p(self.log_dir)
#             # self.summary_writer = SummaryWriter(self.log_dir)
#
#         s_gpus = cfg.GPU_ID.split(',')
#         self.gpus = [int(ix) for ix in s_gpus]
#         self.num_gpus = len(self.gpus)
#         torch.cuda.set_device(self.gpus[0])
#         cudnn.benchmark = True
#
#         self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
#         self.max_epoch = cfg.TRAIN.MAX_EPOCH
#         self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
#
#         self.data_loader = data_loader
#         self.num_batches = len(self.data_loader)
#         self.img_size = img_size
#
#
#     def build_models(self):
#         netG = G_NET()
#         netG.apply(weights_init)
#         netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
#         print(netG)
#
#         netsD = []
#
#         if cfg.TREE.BRANCH_NUM > 0:
#             netsD.append(WD_NET64())
#             # netsD.append(PD_NET64(feature_sizes[3]))
#         if cfg.TREE.BRANCH_NUM > 1:
#             netsD.append(WD_NET128())
#         if cfg.TREE.BRANCH_NUM > 2:
#             netsD.append(WD_NET256())
#
#         for i in range(len(netsD)):
#             netsD[i].apply(weights_init)
#             netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=self.gpus)
#         print('# of netsD', len(netsD))
#
#         count = 0
#         epoch = 0
#         if cfg.TRAIN.PATH != '':
#             if cfg.TRAIN.NET_G != '':
#                 state_dict = torch.load(cfg.TRAIN.NET_G)
#                 netG.load_state_dict(state_dict)
#                 print('load', cfg.TRAIN.NET_G)
#
#                 istart = cfg.TRAIN.NET_G.rfind('_') + 1
#                 iend = cfg.TRAIN.NET_G.rfind('.')
#                 # count = cfg.TRAIN.NET_G[istart:iend]
#                 # count = int(count) + 1
#                 epoch = cfg.TRAIN.NET_G[istart:iend]
#                 epoch = int(epoch)
#             """
#                 for i in range(len(netsD)):
#             netD = netsD[i]
#             torch.save(
#                 netD.state_dict(),
#                 '%s/netD%d.pth' % (model_dir, i))
#              """
#             if cfg.TRAIN.NET_D != '':
#                 for i in range(len(netsD)):
#                     print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
#                     state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i))
#                     netsD[i].load_state_dict(state_dict)
#
#         inception_model = InceptionV3()
#
#         text_encoder = TextEncoder(
#             data_dir='',
#             emb_dim=300,
#             hid_dim=100,
#             z_dim=128,  # 1024
#             # word2vec_file=ckpt_args.word2vec_file,
#             with_attention=True,
#             ingr_enc_type='fc'
#         )
#
#         if cfg.CUDA:
#             netG.cuda()
#             for i in range(len(netsD)):
#                 netsD[i].cuda()
#             inception_model = inception_model.cuda()
#         inception_model.eval()
#
#         return netG, netsD, len(netsD), text_encoder, inception_model, epoch, count
#
#
#
#     def prepare_data(self, data, device='cuda'):
#         title, ing, ins_feat, imgs, w_imgs, _ = data
#         ins_feat = torch.squeeze(ins_feat, dim=1)
#         #print(ins[:1])
#         # txt = torch.cat((ins, ing), 1)  # 16, 1, 512
#         real_vimgs, wrong_vimgs = [], []
#         # imgs = torch.stack(imgs, dim=0)
#         # w_imgs = torch.stack(w_imgs, dim=0)
#         # imgs = imgs[0]
#         # w_imgs = w_imgs[0]
# #         print('len of imgs: ', len(imgs))
# #         b = np.array(imgs)
# #         print('np', imgs.size)
#         for i in range(len(imgs)):
#             real_vimgs.append(imgs[i].to(device))
#             wrong_vimgs.append(w_imgs[i].to(device))
#         # print(real_vimgs[:2])
#         vins_feat = ins_feat.to(device)
#         vtitle = title.clone().detach().to(device)
#         ving = ing.clone().detach().to(device)
# #         print(vtitle.shape)
# #         print(ving.shape)
#
#         # vtitle = [x.to(device) for x in title]
#         # ving = [x.to(device) for x in ing]
#         return vtitle, ving, vins_feat, real_vimgs, wrong_vimgs, imgs
#
#     def gradient_penalty(self, real_samples, fake_samples, labels, netD):
#         # Random weight term for interpolation between real and fake samples
#         alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to('cuda')
#         # Get random interpolation between real and fake samples
#         interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
#         d_interpolates = netD(interpolates, labels)[0]
#         fake = torch.Tensor(real_samples.shape[0]).fill_(1.0).to('cuda')
#         fake.requires_grad = False
#         # Get gradient w.r.t. interpolates
#         gradients = torch.autograd.grad(
#             outputs=d_interpolates,
#             inputs=interpolates,
#             grad_outputs=fake,
#             create_graph=True,
#             retain_graph=True,
#             only_inputs=True,
#         )
#         gradients = gradients[0].view(gradients[0].size(0), -1)
#         gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - 1) ** 2).mean()
#         return gradient_penalty
#
#     def train_Dnet(self, idx, count):
#         flag = 1  # count % 100
#         batch_size = self.real_imgs[0].size(0)  # 1: 24
#         criterion, mu = self.criterion, self.mu  # BCELoss  24, 128
#
#         netD, optD = self.netsD[idx], self.optimizersD[idx]  # Adam
#         real_imgs = self.real_imgs[idx]  # 24,3,64,64
# #         print('real_imgs: ',real_imgs.shape)
#         wrong_imgs = self.wrong_imgs[idx]  # 24,3,64,64
#         fake_imgs = self.fake_imgs[idx]  # 24,3,64,64
#         # get efficient net features
#         netD.zero_grad()
#         # Forward
#         real_labels = self.real_labels[:batch_size].to('cuda')  # 24
#         fake_labels = self.fake_labels[:batch_size].to('cuda')  # 24
#         # logits:[[conditional], [unconditional]]
#         real_logits = netD(real_imgs, mu.detach())  # list:2
#         errD_real = torch.mean(real_logits[0])
#
#         fake_logits = netD(fake_imgs.detach(), mu.detach())  # list:2
#         errD_fake = torch.mean(fake_logits[0])
#
#         wrong_logits = netD(wrong_imgs, mu.detach())  # list:2
#         errD_wrong = torch.mean(wrong_logits[0])
#
#         gp = self.gradient_penalty(real_imgs.detach(), fake_imgs.detach(), mu.detach(), netD)
#
#         # Unconditional loss
#
#         if len(real_logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
#             errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
#                                criterion(real_logits[1], real_labels)
#
#             errD_wrong_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
#                                 criterion(wrong_logits[1], real_labels)
#             errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
#                                criterion(fake_logits[1], fake_labels)
#             #
#             errD_cond = errD_real + errD_wrong + errD_fake
#             errD_uncond = errD_real_uncond + errD_fake_uncond + errD_wrong_uncond
#             errD_real = errD_real + errD_real_uncond  # float
#             errD_wrong = errD_wrong + errD_wrong_uncond
#             errD_fake = errD_fake + errD_fake_uncond
#             #
#
#             errD = errD_real + errD_wrong + errD_fake
#             wandb.log({
#                 f'errD_cond{idx}': errD_cond,
#                 f'errD_uncond{idx}': errD_uncond,
#                 f'errD{idx}': errD,
#                 f'batch_idx': count,  # batch_idx
#             })
#         else:
#             # lmbda_gp = 10
#             errD = 0.5 * (errD_fake + errD_wrong) - errD_real + 10 * gp
# #             was_loss = 0.5 * (errD_fake + errD_wrong) + errD_real
#             wandb.log({
#                 f'errD{idx}': errD,
# #                 f'Was_loss{idx}': was_loss,
#                 f'batch_idx': count,  # batch_idx
#             })
#
# #         was_loss.backward()
#         # backward
#         errD.backward()
#         # update parameters
#         optD.step()
#         # log
#         # if flag == 0:
#         #     self.summary_writer.add_scalar('D_loss%d'%idx, errD.item(),count)
#
#         return errD
#
#     def train_Gnet(self, count):  # add MSE loss
#         self.netG.zero_grad()
#         errG_total = 0
#         flag = 1  # count % 100
#         batch_size = self.real_imgs[0].size(0)
#         criterion, mu, logvar = self.criterion, self.mu, self.logvar
#         real_labels = self.real_labels[:batch_size].to('cuda')
#         for i in range(self.num_Ds):
#             outputs = self.netsD[i](self.fake_imgs[i], mu)
#             # mse_loss = nn.MSE_loss(outputs[0], )
#             errG = - torch.mean(outputs[0])
#
#             if len(outputs) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
#                 errG_patch = cfg.TRAIN.COEFF.UNCOND_LOSS * \
#                              criterion(outputs[1], real_labels)
#                 errG = errG + errG_patch
#             errG_total = errG_total + errG
#             # if flag == 0:
#             #     self.summary_writer.add_scalar('G_loss%d' % i, errG.item(), count)
#             wandb.log({
#                 f'errG': errG,
#                 f'batch_idx': count,  # batch_idx
#             })
#
#         # Compute color consistency losses
#         if cfg.TRAIN.COEFF.COLOR_LOSS > 0:
#             if self.num_Ds > 1:
#                 mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-1])
#                 mu2, covariance2 = \
#                     compute_mean_covariance(self.fake_imgs[-2].detach())
#                 like_mu2 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
#                 like_cov2 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
#                             nn.MSELoss()(covariance1, covariance2)
#                 errG_total = errG_total + like_mu2 + like_cov2
#                 # if flag == 0:
#                 #
#             if self.num_Ds > 2:
#                 mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-2])
#                 mu2, covariance2 = \
#                     compute_mean_covariance(self.fake_imgs[-3].detach())
#                 like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
#                 like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
#                             nn.MSELoss()(covariance1, covariance2)
#                 errG_total = errG_total + like_mu1 + like_cov1
#                 # if flag == 0:
#
#
#         kl_loss = KL_loss(mu, logvar) * cfg.TRAIN.COEFF.KL
# #         errG_total = errG_total + kl_loss
#         errG_total.backward()
#         self.optimizerG.step()
#         return kl_loss, errG_total
#
#     def train(self):
#         self.netG, self.netsD, self.num_Ds, \
#             self.text_encoder, self.inception_model, epoch, count = self.build_models()
#
#         avg_param_G = copy_G_params(self.netG)
#
#         self.optimizerG, self.optimizersD = \
#             define_optimizers(self.netG, self.netsD)
#
#         self.criterion = nn.BCELoss()
#
#         # self.real_labels = \
#         #     Variable(torch.FloatTensor(self.batch_size).fill_(1))
#         # self.fake_labels = \
#         #     Variable(torch.FloatTensor(self.batch_size).fill_(0))
#
#         # Prepare labels
#         if cfg.LABELS == 'original':
#             self.real_labels = torch.FloatTensor(self.batch_size).fill_(
#                 1)  # (torch.FloatTensor(args.batch_size).uniform_() < 0.9).float() #
#             self.fake_labels = torch.FloatTensor(self.batch_size).fill_(
#                 0)  # (torch.FloatTensor(args.batch_size).uniform_() > 0.9).float() #
#         elif cfg.LABELS == 'R-smooth':
#             self.real_labels = torch.FloatTensor(self.batch_size).fill_(1) - (
#                     torch.FloatTensor(self.batch_size).uniform_() * 0.1)
#             self.fake_labels = (torch.FloatTensor(self.batch_size).uniform_() * 0.1)
#         # elif args.labels == 'R-flip':
#         #     real_labels = (torch.FloatTensor(args.batch_size).uniform_() < 0.9).float()  #
#         #     fake_labels = (torch.FloatTensor(args.batch_size).uniform_() > 0.9).float()  #
#         # elif args.labels == 'R-flip-smooth':
#         #     real_labels = torch.abs((torch.FloatTensor(args.batch_size).uniform_() > 0.9).float() - (
#         #             torch.FloatTensor(args.batch_size).fill_(1) - (
#         #             torch.FloatTensor(args.batch_size).uniform_() * 0.1)))
#         #     fake_labels = torch.abs((torch.FloatTensor(args.batch_size).uniform_() > 0.9).float() - (
#         #             torch.FloatTensor(args.batch_size).uniform_() * 0.1))
#
#         self.gradient_one = torch.FloatTensor([1.0])
#         self.gradient_half = torch.FloatTensor([0.5])
#
#         nz = cfg.GAN.Z_DIM
#         noise = Variable(torch.FloatTensor(self.batch_size, nz))
#         fixed_noise = \
#             Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))
#
#         if cfg.CUDA:
#             self.criterion.cuda()
#             self.real_labels = self.real_labels.cuda()
#             self.fake_labels = self.fake_labels.cuda()
#             self.gradient_one = self.gradient_one.cuda()
#             self.gradient_half = self.gradient_half.cuda()
#             noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
#
#         predictions = []
#
#         count = epoch * (self.num_batches) + 1
#         start_epoch = epoch + 1
#         # start_epoch = start_count // (self.num_batches)
#         print('Start from count ', count)
#         print('Start from epoch', start_epoch)
#         print('Start training...')
#         for epoch in range(start_epoch, self.max_epoch):
#             # print('Epoch ', epoch)
#             start_t = time.time()
#             data_iter = iter(self.data_loader)
#             step = 0
#             while step < self.num_batches:
#                 # print(count)
#                 #######################################################
#                 # (0) Prepare training data
#                 ######################################################
#                 # print("Preparing training data...")
#                 data = next(data_iter)
#
#                 title, ing, ins_feat, self.real_imgs, self.wrong_imgs, self.imgs_tcpu = self.prepare_data(data)
#                 # self.imgs_tcpu: real images in cpu
#                 with torch.no_grad():
#                     txt_feat = self.text_encoder([title, ing])
# #                 print(txt_feat.shape)
# #                 print(ins_feat.shape)
#                 self.txt_embedding = torch.cat([txt_feat, ins_feat], dim=1)
#                 #######################################################
#                 # (1) Generate fake images
#                 ######################################################
#                 # print("Generating fake imgs...")
#                 noise.data.normal_(0, 1)
#                 self.fake_imgs, self.mu, self.logvar = \
#                     self.netG(noise, self.txt_embedding)
#
#                 #######################################################
#                 # (2) Update D network
#                 ######################################################
#                 # print("Updating networks...")
#                 errD_total = 0
#                 was_total = 0
#                 for i in range(self.num_Ds):
#                     errD = self.train_Dnet(i, count)
#                     # errD = self.train_Dnet_projected(i, count)
#                     errD_total += errD
# #                     was_total += was_loss
#
#                 #######################################################
#                 # (3) Update G network: maximize log(D(G(z)))
#                 ######################################################
#                 kl_loss, errG_total = self.train_Gnet(count)
#                 for p, avg_p in zip(self.netG.parameters(), avg_param_G):
#                     avg_p.mul_(0.999).add_(p.data, alpha=0.001)
#
#                 # for inception score
#                 pred_is = self.inception_model(self.fake_imgs[-1].detach(), 'is')
#
#
#                 predictions.append(pred.data.cpu().numpy())
#
#                 wandb.log({
#                     f'D_loss': errD_total,
#                     f'G_loss': errG_total,
#                     f'KL_loss': kl_loss,
# #                     f'Was_loss': was_total,
#                     f'batch_idx': count,  # batch_idx
#                 })
#
#                 # Compute inception score
#                 if len(predictions) > 500:
#                     predictions = np.concatenate(predictions, 0)
#                     mean, std = compute_inception_score(predictions, 10)
#                     #
#                     mean_nlpp, std_nlpp = \
#                         negative_log_posterior_probability(predictions, 10)
#
#                     wandb.log({
#                     f'Inception_mean': mean,
#                     f'NLPP_mean': mean_nlpp,
#                     f'batch_idx': count,
#                     })
#                     #
#                     predictions = []
#                 step = step + 1
#                 count = count + 1
#
#             end_t = time.time()
#             if epoch % 5 == 0:
#                 save_model(self.netG, avg_param_G, self.netsD, epoch, self.model_dir)
#                 # Save images
#                 backup_para = copy_G_params(self.netG)
#                 load_params(self.netG, avg_param_G)
#                 #
#                 self.fake_imgs, _, _ = \
#                     self.netG(fixed_noise, self.txt_embedding)
#                 save_img_results(self.imgs_tcpu, self.fake_imgs, self.num_Ds,
#                                  count, epoch, self.image_dir)
#                 #
#                 load_params(self.netG, backup_para)
#
#             print('''[%d/%d][%d]
#                          Loss_D: %.2f Loss_G: %.2f Loss_KL: %.2f Time: %.2fs
#                       '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
#                   % (epoch, self.max_epoch, self.num_batches,
#                      errD_total.item(), errG_total.item(),
#                      kl_loss.item(), end_t - start_t))
#             # if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
#             #     self.save_model(netG, avg_param_G, netsD, epoch)
#
#         save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
#
#     def save_superimages(self, images_list, filenames,
#                          save_dir, split_dir, imsize):
#         batch_size = images_list[0].size(0)
#         num_sentences = len(images_list)
#         for i in range(batch_size):
#             s_tmp = '%s/super/%s/%s' % \
#                     (save_dir, split_dir, filenames[i])
#             folder = s_tmp[:s_tmp.rfind('/')]
#             if not os.path.isdir(folder):
#                 print('Make a new folder: ', folder)
#                 mkdir_p(folder)
#             #
#             savename = '%s_%d.png' % (s_tmp, imsize)
#             super_img = []
#             for j in range(num_sentences):
#                 img = images_list[j][i]
#                 # print(img.size())
#                 img = img.view(1, 3, imsize, imsize)
#                 # print(img.size())
#                 super_img.append(img)
#                 # break
#             super_img = torch.cat(super_img, 0)
#             vutils.save_image(super_img, savename, nrow=10, normalize=True)
#
#     def save_singleimages(self, images, filenames,
#                           save_dir, split_dir, sentenceID, imsize):
#         for i in range(images.size(0)):
#             s_tmp = '%s/single_samples/%s/%s' % \
#                     (save_dir, split_dir, filenames[i])
#             folder = s_tmp[:s_tmp.rfind('/')]
#             if not os.path.isdir(folder):
#                 print('Make a new folder: ', folder)
#                 mkdir_p(folder)
#
#             fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
#             # range from [-1, 1] to [0, 255]
#             img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
#             ndarr = img.permute(1, 2, 0).data.cpu().numpy()
#             im = Image.fromarray(ndarr)
#             im.save(fullpath)
#
#     def evaluate(self, split_dir):
#         if cfg.TRAIN.NET_G == '':
#             print('Error: the path for morels is not found!')
#         else:
#             # Build and load the generator
#             if split_dir == 'test':
#                 split_dir = 'valid'
#             netG = G_NET()
#             netG.apply(weights_init)
#             netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
#             print(netG)
#             # state_dict = torch.load(cfg.TRAIN.NET_G)
#             state_dict = \
#                 torch.load(cfg.TRAIN.NET_G,
#                            map_location=lambda storage, loc: storage)
#             netG.load_state_dict(state_dict)
#             print('Load ', cfg.TRAIN.NET_G)
#
#             # the path to save generated images
#             s_tmp = cfg.TRAIN.NET_G
#             istart = s_tmp.rfind('_') + 1
#             iend = s_tmp.rfind('.')
#             iteration = int(s_tmp[istart:iend])
#             s_tmp = s_tmp[:s_tmp.rfind('/')]
#             save_dir = '%s/iteration%d' % (s_tmp, iteration)
#
#             nz = cfg.GAN.Z_DIM
#             noise = Variable(torch.FloatTensor(self.batch_size, nz))
#             if cfg.CUDA:
#                 netG.cuda()
#                 noise = noise.cuda()
#
#             # switch to evaluate mode
#             netG.eval()
#             for step, data in enumerate(self.data_loader, 0):
#                 imgs, t_embeddings, filenames = data
#                 if cfg.CUDA:
#                     t_embeddings = Variable(t_embeddings).cuda()
#                 else:
#                     t_embeddings = Variable(t_embeddings)
#                 # print(t_embeddings[:, 0, :], t_embeddings.size(1))
#
#                 embedding_dim = t_embeddings.size(1)
#                 batch_size = imgs[0].size(0)
#                 noise.data.resize_(batch_size, nz)
#                 noise.data.normal_(0, 1)
#
#                 fake_img_list = []
#                 for i in range(embedding_dim):
#                     fake_imgs, _, _ = netG(noise, t_embeddings[:, i, :])
#                     if cfg.TEST.B_EXAMPLE:
#                         # fake_img_list.append(fake_imgs[0].data.cpu())
#                         # fake_img_list.append(fake_imgs[1].data.cpu())
#                         fake_img_list.append(fake_imgs[2].data.cpu())
#                     else:
#                         self.save_singleimages(fake_imgs[-1], filenames,
#                                                save_dir, split_dir, i, 256)
#                         # self.save_singleimages(fake_imgs[-2], filenames,
#                         #                        save_dir, split_dir, i, 128)
#                         # self.save_singleimages(fake_imgs[-3], filenames,
#                         #                        save_dir, split_dir, i, 64)
#                     # break
#                 if cfg.TEST.B_EXAMPLE:
#                     # self.save_superimages(fake_img_list, filenames,
#                     #                       save_dir, split_dir, 64)
#                     # self.save_superimages(fake_img_list, filenames,
#                     #                       save_dir, split_dir, 128)
#                     self.save_superimages(fake_img_list, filenames,
#                                           save_dir, split_dir, 256)
#
