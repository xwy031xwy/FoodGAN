import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.optim import Adam
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.cuda


# ############# For Compute inception score ############
# Besides the inception score computed by pretrained model, especially for fine-grained datasets (such as birds, bedroom),
#  it is also good to compute inception score using fine-tuned model and manually examine the image quality.

class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3(pretrained=True).to('cuda')  # score
        #         url = 'http://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        #         # print(next(model.parameters()).data)
        #         state_dict = \
        #             model_zoo.load_url(url, map_location=lambda storage,  loc:storage)
        # Download pretrained valuation model from url
        # Local model 
#         state_path = "inception_v3_google-1a9a5a14.pth"
#         state_dict = torch.load(state_path)
#         self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        #         print('Load pretrained model from', url)
#         print('Get pretrained model: ', state_path)

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        x = input * 0.5 + 0.5
        # ImageNet という画像分類データセットの RGB の平均と標準偏差:
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
#         x[:, 0] = (x[:, 0] - 0.485) / 0.229
#         x[:, 1] = (x[:, 1] - 0.456) / 0.224
#         x[:, 2] = (x[:, 2] - 0.406) / 0.225

        # --> fixed-size input: batch x 3 x 299 x 299
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # 299 x 299 x 3
        x = self.model(x)
        x = nn.Softmax(dim=1)(x)
        return x


class InceptionFID(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
       
        super(InceptionFID, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        
        
        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
    

    
    
# ############ unit #############
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()  # 親クラスを示すためのメソッド

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'  # 条件をテストするデバッグ支援ツール
        # アサーションの条件がFalseと評価された場合はAssertionError例外が送出され、必要に応じてエラーメッセージ
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])  # F = function


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# ############ G networks ############
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# 残差ブロック
class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # 残差を計算
        return out


# Conditioning Argumentation
class CA_NET(nn.Module):
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):  # batch size, 1024
        # 対角ガウス
        # ガウス分布の平均と対数分散に（数値の安定性のため）
        # x = self.relu(self.fc(text_embedding))  # batch size, 256
        x = self.relu(text_embedding)
        mu = x[:, :self.c_dim]  # batch, c_dim = 128
        logvar = x[:, self.c_dim:]  # batch, 128
        return mu, logvar

    def reparametrize(self, mu, logvar):
        # 単位ガウスからサンプリング * std + 平均値mul
        # -> 勾配がサンプルからパラメータに伝播される
        std = logvar.mul(0.5).exp_().to(mu.device)  # 24,128
        # Generate Gaussian noise with the same size as std
        if cfg.CUDA:
            eps = torch.randn(std.size(), device='cuda', dtype=torch.float)  # 24, 128
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps.to(mu.device))
        return eps.mul(std).add_(mu)  # mul a:eps.mul(std) b: mu

    def forward(self, text_embedding):
        # self.txt_embedding = self.prepare_data(data)
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf  # 64
        if cfg.GAN.B_CONDITION:
            self.in_dim = cfg.GAN.Z_DIM + cfg.GAN.EMBEDDING_DIM  # 228
        else:
            self.in_dim = cfg.GAN.Z_DIM
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim  # 228
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU()
        )

        self.upsample1 = upBlock(ngf, ngf // 2)  # //: 整数除算（切り捨て除算）
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

        """
        def upBlock(in_planes, out_planes):
            block = nn.Squential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes*2),
            nn.BatchNorm2d(out_planes*2),
            GLU()
            )
            return block
        """

    def forward(self, z_code, c_code=None):  # z: 24, 100 c: 24, 128
        if cfg.GAN.B_CONDITION and c_code is not None:
            in_code = torch.cat((c_code, z_code), 1)  # 24, 228
        else:
            in_code = z_code
        # state size 16ngf * 4 * 4
        out_code = self.fc(in_code)  #
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # 相当于numpy中的reshape，重新定义矩阵的形状
        # view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
        # state size 8ngf * 8 * 8
        out_code = self.upsample1(out_code)
        # state size 4ngf * 16 * 16
        out_code = self.upsample2(out_code)
        # state size 2ngf * 32 * 32
        out_code = self.upsample3(out_code)
        # state size ngf * 64 * 64
        out_code = self.upsample4(out_code)

        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            self.ef_dim = cfg.GAN.EMBEDDING_DIM  # 128
        else:
            self.ef_dim = cfg.GAN.Z_DIM
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim  # 64
        efg = self.ef_dim  # 128
        #         print('ngf', ngf)
        #         print('efg', efg)

        self.jointConv = Block3x3_relu(ngf + efg, ngf)  # 192, 64
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        #         print('h_code', h_code.shape)
        #         print('c_code', c_code.shape)
        # h_code torch.Size([128, 64, 64, 64])
        # c_code torch.Size([128, 128])
        s_size = h_code.size(2)
        #         print('s_size', s_size)
        # s_size 64
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        # state size (ngf+efg) * in_size * in_size
        h_c_code = torch.cat((c_code, h_code), 1)
        # h_c_code_cat torch.Size([128, 192, 64, 64])
        #         print('h_c_code_cat', h_c_code.shape)
        # state size ngf * in_size * in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        # state size ngf/2 * 2in_size * 2in_size
        out_code = self.upsample(out_code)

        return out_code


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.g_dim = cfg.GAN.GF_DIM  # 64
        self.define_module()
        self.ca_net = CA_NET()

    def define_module(self):
        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(self.g_dim * 16)
            self.img_net1 = GET_IMAGE_G(self.g_dim)
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(self.g_dim)
            self.img_net2 = GET_IMAGE_G(self.g_dim // 2)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(self.g_dim // 2)
            self.img_net3 = GET_IMAGE_G(self.g_dim // 4)
        if cfg.TREE.BRANCH_NUM > 3:
            self.h_net4 = NEXT_STAGE_G(self.g_dim // 4, num_residual=1)
            self.img_net4 = GET_IMAGE_G(self.g_dim // 8)
        if cfg.TREE.BRANCH_NUM > 4:
            self.h_net4 = NEXT_STAGE_G(self.g_dim // 8, num_residual=1)
            self.img_net4 = GET_IMAGE_G(self.g_dim // 16)

    def forward(self, z_code, text_embedding=None):
        if cfg.GAN.B_CONDITION and text_embedding is not None:
            c_code, mu, logvar = self.ca_net(text_embedding)
        else:
            c_code, mu, logvar = z_code, None, None
        fake_imgs = []
        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_imgs1 = self.img_net1(h_code1)
            fake_imgs.append(fake_imgs1),
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            fake_imgs2 = self.img_net2(h_code2)
            fake_imgs.append(fake_imgs2)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            fake_imgs3 = self.img_net3(h_code3)
            fake_imgs.append(fake_imgs3)
        if cfg.TREE.BRANCH_NUM > 3:
            h_code4 = self.h_net4(h_code3, c_code)
            fake_imgs4 = self.img_net4(h_code4)
            fake_imgs.append(fake_imgs4)

        return fake_imgs, mu, logvar


# ############ D networks ############
def Block3x3_leakReLu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downscale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downscale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(

        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        # 第1引数はその入力のチャネル数,第2引数は畳み込み後のチャネル数,第3引数は畳み込みをするための正方形フィルタ(カーネル)の1辺のサイズである.
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size, ndf * in_size/2 * in_size/2
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size, 2ndf * in_size/4 * in_size/4
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size, 4ndf * in_size/8 * in_size/8
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size, 8ndf * in_size/16 * in_size/16
    )
    return encode_img


# For 64 * 64 images
class D_NET64(nn.Module):
    def __init__(self):
        super(D_NET64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM  # 64
        self.ef_dim = cfg.GAN.EMBEDDING_DIM  # 512
        self.define_module()

    def define_module(self):
        ndf = self.df_dim  # number of discriminator features  64
        efg = self.ef_dim  # embedding features of gan
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.logits_sn = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)),
            nn.Sigmoid()
        )
        #self.conv2d = spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
#         self.sigmoid = nn.Sigmoid()
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
        )

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakReLu(ndf * 8 + efg, ndf * 8)  #

            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid()
            )
            self.uncond_logits_sn = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):  # 24,128
#         print("x_var: ", x_var.shape)
#         print("c_code: ", c_code.shape)
        x_code = self.img_code_s16(x_var)  # 24,3,64,64 -> 24,512,4,4
        # print("x_code(BS, 512, 4, 4): ", x_code.shape)
        # x_var = img
        # x_code: img code
        # c_code: conditional code
        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)  # 24,128,1,1
            # print("c_code1(BS, 128, 1, 1): ", c_code.shape)
            c_code = c_code.repeat(1, 1, 4, 4)  # 24,128,4,4
            # print("c_code2(BS, 128, 4, 4): ", c_code.shape)
            # repeat(要素や配列, 繰り返し回数, (繰り返す方向) )
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)  # 24, 640,4,4
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)  # 24, 512, 4,4
        else:
            h_c_code = x_code
        
#         pre_out = self.conv2d(h_c_code) # for the BCR
#         output = self.sigmoid(pre_out)
        
        output = self.logits(h_c_code)  # 24,1,1,1
        
        if cfg.GAN.B_CONDITION:
            # out_uncond = self.uncond_logits(x_code)  # 24,1,1,1
            out_uncond = self.uncond_logits_sn(x_code)  # 24,1,1,1
            # return [output.view(-1), out_uncond.view(-1), pre_out.view(-1)]
            return [output.view(-1), out_uncond.view(-1)]
        else:
            # return [output.view(-1), pre_out.view(-1)]
            return [output.view(-1)]


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self):
        super(D_NET128, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakReLu(ndf * 16, ndf * 8)
        
        #self.conv2d = spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
        # self.sigmoid = nn.Sigmoid()
        self.logits = nn.Sequential(
            (nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)),
            nn.Sigmoid()
        )

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakReLu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                (nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        #pre_out = self.conv2d(h_c_code)
        #output = self.sigmoid(pre_out)
        output = self.logits(h_c_code)  # 24,1,1,1
        pre_out = 0
        if cfg.GAN.B_CONDITION:
            # out_uncond = self.uncond_logits(x_code)  # 24,1,1,1
            out_uncond = self.uncond_logits(x_code)  # 24,1,1,1
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self):
        super(D_NET256, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakReLu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakReLu(ndf * 16, ndf * 8)

#         self.conv2d = spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
#         self.sigmoid = nn.Sigmoid()
        self.logits = nn.Sequential(
            (nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakReLu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                (nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

#         pre_out = self.conv2d(h_c_code)
#         output = self.sigmoid(pre_out)
        output = self.logits(h_c_code)  # 24,1,1,1
        pre_out = 0
        if cfg.GAN.B_CONDITION:
            # out_uncond = self.uncond_logits(x_code)  # 24,1,1,1
            out_uncond = self.uncond_logits(x_code)  # 24,1,1,1
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]




def kaiming_init(module):
    classname = module.__class__.__name__  # 名前
    # Convなら、初期化
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

    """
    Implementation for the proposed Cross-Scale Mixing.
    """


class CSM(nn.Module):

    def __init__(self, channels, conv3_out_channels):
        super(CSM, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, 1, 1).to("cuda")
        self.conv3 = nn.Conv2d(channels, conv3_out_channels, 3, 1, 1).to("cuda")

        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.conv3.parameters():
            param.requires_grad = False

        self.apply(kaiming_init)

    def forward(self, high_res, low_res=None):
        batch, channels, width, height = high_res.size()
        if low_res is None:
            # high_res_flatten = rearrange(high_res, "b c h w -> b c (h w)")
            high_res_flatten = high_res.view(batch, channels, width * height)
            high_res = self.conv1(high_res_flatten)
            high_res = high_res.view(batch, channels, width, height)
            high_res = self.conv3(high_res)
            high_res = F.interpolate(high_res, scale_factor=2., mode="bilinear")
            return high_res
        else:
            high_res_flatten = high_res.view(batch, channels, width * height)
            high_res = self.conv1(high_res_flatten)
            high_res = high_res.view(batch, channels, width, height)
            high_res = torch.add(high_res, low_res)
            high_res = self.conv3(high_res)
            # 对其进行双线性插值，将其尺寸增大两倍
            high_res = F.interpolate(high_res, scale_factor=2., mode="bilinear")
            return high_res


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, 4, 2, 1)
        self.bn = nn.BatchNorm2d(c_out)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.leaky_relu(x)



class MultiScaleDiscriminator(nn.Module):
    def __init__(self, channels, stage):
        super(MultiScaleDiscriminator, self).__init__()
        # self.head_conv = spectral_norm(nn.Conv2d(512, 1, 3, 1, 1))
        self.head_conv = spectral_norm(nn.Conv2d(64 * stage, 1, 3, 1, 1))  # input, 1
        # layers = []
        # if l == 1:
        #     layers.append(DownBlock(c_in, 64))
        #     layers.append(DownBlock(64, 128))
        #     layers.append(DownBlock(128, 256))
        #     layers.append(DownBlock(256, 512))
        # elif l == 2:
        #     layers.append(DownBlock(c_in, 128))
        #     layers.append(DownBlock(128, 256))
        #     layers.append(DownBlock(256, 512))
        # elif l == 3:
        #     layers.append(DownBlock(c_in, 256))
        #     layers.append(DownBlock(256, 512))
        # else:
        #     layers.append(DownBlock(c_in, 512))
        layers = [DownBlock(channels, 64 * stage)] + [DownBlock(64 * i, 64 * i * 2) for i in [1, 2][stage - 1:]]
        # layers = [DownBlock(channels, 64 * [1, 2, 4, 8][l - 1])] + [DownBlock(64 * i, 64 * i * 2) for i in [1, 2, 4][l - 1:]]
        self.model = nn.Sequential(*layers)
        self.optim = Adam(self.model.parameters(), lr=0.0002, betas=(0, 0.99))

    def forward(self, x):
        x = self.model(x)
        return self.head_conv(x)


class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code
    
# For 64 * 64 images
class WD_NET64(nn.Module):
    def __init__(self):
        super(WD_NET64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM  # 64
        self.ef_dim = cfg.GAN.EMBEDDING_DIM  # 512
        self.define_module()

    def define_module(self):
        ndf = self.df_dim  # number of discriminator features  64


        efg = self.ef_dim  # embedding features of gan
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.logits_sn = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
        )


        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
        )

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakReLu(ndf * 8 + efg, ndf * 8)  #

            self.uncond_logits_sn = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)))

    def forward(self, x_var, c_code=None):  # 24,128
#         print("x_var: ", x_var.shape)
#         print("c_code: ", c_code.shape)
        x_code = self.img_code_s16(x_var)  # 24,3,64,64 -> 24,512,4,4
        # print("x_code(BS, 512, 4, 4): ", x_code.shape)
        # x_var = img
        # x_code: img code
        # c_code: conditional code
        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)  # 24,128,1,1
            # print("c_code1(BS, 128, 1, 1): ", c_code.shape)
            c_code = c_code.repeat(1, 1, 4, 4)  # 24,128,4,4
            # print("c_code2(BS, 128, 4, 4): ", c_code.shape)
            # repeat(要素や配列, 繰り返し回数, (繰り返す方向) )
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)  # 24, 640,4,4
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)  # 24, 512, 4,4
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)  # 24,1,1,1
        if cfg.GAN.B_CONDITION:
            # out_uncond = self.uncond_logits(x_code)  # 24,1,1,1
            out_uncond = self.uncond_logits_sn(x_code)  # 24,1,1,1
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


# For 128 x 128 images
class WD_NET128(nn.Module):
    def __init__(self):
        super(WD_NET128, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakReLu(ndf * 16, ndf * 8)

        self.logits_sn = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
        )
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)
        )

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakReLu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)))

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


# For 256 x 256 images
class WD_NET256(nn.Module):
    def __init__(self):
        super(WD_NET256, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakReLu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakReLu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
        self.logits_sn = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)))

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakReLu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)))

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]
