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
import copy
import numpy as np
from scipy import linalg

# ############# For Compute inception score ############
# Besides the inception score computed by pretrained model, especially for fine-grained datasets (such as birds, bedroom),
#  it is also good to compute inception score using fine-tuned model and manually examine the image quality.



class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    # Index of default block of inception to return,
    # corresponds to output of final average pooling

    def __init__(self, requires_grad=False):
        
        super(InceptionV3, self).__init__()
        dropout = float(0.5)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, 1000).cuda() # num_classes
        

        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        inception = models.inception_v3(weights=weights).cuda()
        # make sure every layer is same as original
        block = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            inception.maxpool1,
#             nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            inception.maxpool2,
#             nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            inception.avgpool
#             nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ]
        self.model = nn.Sequential(*block)
        
        self.dropout = inception.dropout
        self.fc = inception.fc
        
        for param in self.parameters():
            param.requires_grad = requires_grad
    def transform_input(self, x):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x
    
    def forward(self, input):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        """

        x = input * 0.5 + 0.5
        # ImageNet という画像分類データセットの RGB の平均と標準偏差:
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        # inceptionv3  def _transform_input(self, x: Tensor) -> Tensor:
#         x[:, 0] = (x[:, 0] - 0.485) / 0.229
#         x[:, 1] = (x[:, 1] - 0.456) / 0.224
#         x[:, 2] = (x[:, 2] - 0.406) / 0.225
        
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        x = self.transform_input(x)
        # torch.Size([16, 3, 299, 299])
        pred_fid = self.model(x)
#         print(pred_fid[0].shape) 2048,1 ,1
        tmp = copy.deepcopy(pred_fid)
        
        # N x 2048 x 1 x 1
        tmp = self.dropout(tmp)
        # N x 2048 x 1 x 1
        tmp = torch.flatten(tmp, 1)
        # N x 2048
        tmp = self.fc(tmp)
        # N x 1000 (num_classes)
        pred_is = nn.Softmax(dim=1)(tmp)  # inceptionv3 classifer
        # BS x 1000 (num_classes)
        
        return pred_is, pred_fid
    
    
def compute_inception_score(predictions, num_splits=1):
    # print('predictions', prediction.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
             (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def calculate_activation_statistics(images, model, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    if cuda:
        batch=images.cuda()
    else:
        batch=images
    
    pred = model(batch)
    # print('pred in cas func0:', len(pred))  # 1
    pred = pred[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    """
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    """
    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)

    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    return mu, sigma

def calculate_activation_statistics(act, dims=2048, cuda=False):

#     if cfg.DEBUG == 1:
#         print('act in cas func:', act.shape)
#         print(act)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
#     if cfg.DEBUG == 1:
#         print('mu in cas func:', mu.shape)# (2048,)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    eps = 1e-6
    sigma1 += eps * np.eye(sigma1.shape[0])
    sigma2 += eps * np.eye(sigma2.shape[0])
    
    diff = mu1 - mu2

    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_fretchet_pre(images_real, images_fake, model):
    mu_1, std_1 = calculate_activation_statistics(images_real,model,cuda=True)
    mu_2, std_2 = calculate_activation_statistics(images_fake,model,cuda=True)
    
    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value

def calculate_fretchet(preds_real, preds_fake):
    mu_1, std_1 = calculate_activation_statistics(preds_real,cuda=True)
    mu_2, std_2 = calculate_activation_statistics(preds_fake,cuda=True)
    
    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value