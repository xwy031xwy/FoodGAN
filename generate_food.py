import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from torchvision import transforms
import pickle
import numpy as np
import os
import json
from gensim.models.keyedvectors import KeyedVectors
from PIL import Image
import sys
from transformers import DistilBertTokenizer, DistilBertModel
from word2vec.get_wordvec import load_recipes, get_title_wordvec, get_ingredients_wordvec, get_instructions_wordvec
from model import G_NET
from text_encoder import TextEncoder
import matplotlib.pyplot as plt
import text_encoder
from miscc.config import cfg
from torch.autograd import Variable

from PIL import Image
import torch


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


def build_models(netG_path):
    netG = G_NET()
    netG.eval()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG)
    # print(netG)

    state_dict = torch.load(netG_path)
    netG.load_state_dict(state_dict)
    print('load', netG_path)

    text_encoder = TextEncoder(
        data_dir='word2vec',
        emb_dim=300,
        hid_dim=100,
        z_dim=256,  # 1024
        # word2vec_file=ckpt_args.word2vec_file,
        with_attention=True,
        ingr_enc_type='fc'
    )

    text_encoder.eval()

    if cfg.CUDA:
        netG.cuda()

    return netG, text_encoder


def get_embedding(text, tokenizer, text_encoder):
    token = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = text_encoder(**token)
    features = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size768)
    return features


word2vec_file = 'word2vec/word2vec_recipes.bin'
vocab_ingrs_file = 'word2vec/list_of_merged_ingredients.txt'

wv = KeyedVectors.load(word2vec_file, mmap='r')
w2i = {w: i + 2 for i, w in enumerate(wv.index_to_key)}
w2i['<other>'] = 1
w2i = w2i

with open(vocab_ingrs_file, 'r') as f:
    vocab_ingrs = f.read().strip().split('\n')
    ingr2i = {ingr: i for i, ingr in enumerate(vocab_ingrs)}
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
ins_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
# 创建一个线性层
linear_ing = nn.Linear(768, 128)
linear_ins = nn.Linear(768, 256)

netG_path = 'models/netG_saladall_300.pth'

netG, text_encoder = build_models(netG_path)

# Prepare your recipe
recipe = \
    {'title': "Greek Salad",
     'ingredients': ["Cucumber", "green bell pepper", "tomato", "red onion", "Kalamata olives", "fresh mint leaves"],
     'instructions': ["Make the dressing: In a small bowl, whisk together the olive oil, vinegar, garlic, oregano, mustard, salt, and several grinds of pepper.",
                      "On a large platter, arrange the cucumber, green pepper, feta cheese, red onions, and olives.",
                      "Drizzle with the dressing and very gently toss. ",
                      "Sprinkle with a few generous pinches of oregano and top with the mint leaves. Season to taste and serve."]}

"""  
{'title': "Mummy Cupcakes",
          'ingredients': ["cake_mix",
                          "Oreo_cookies",
                          "pudding_mix",
                          "milk",
                          "powdered_sugar",
                          "Cool_Whip",
                          "chocolate_chips"],
          'instructions': [
              "Heat oven to 350F .",
              "Prepare cake batter as directed on package ; stir in cookies .",
              "Spoon into 24 paper-lined muffin cups .",
              "Bake 20 to 25 min .",
              "or until toothpick inserted in centres comes out clean .",
              "Cool cupcakes in pans 10 min .",
              "; remove to wire racks .",
              "Beat pudding_mix , milk and sugar in large bowl whisk 2 min . , stir in Cool_Whip .",
              "Spoon into pastry bag fitted with basket-weave tip .",
              "Pipe pudding_mixture onto tops of cupcakes .",
              "Add chocolate_chips for the eyes .",
              "Keep refrigerated ."]
          }
"""
instructions = recipe['instructions']
if len(instructions) > 0:
    instructions = ' '.join(instructions)
title, n_words_in_title = get_title_wordvec(recipe, w2i)  # np.int [max_len]  text to index
ingredients, n_ingrs = get_ingredients_wordvec(recipe, w2i, ingr2i)  # np.int [max_len]
title = torch.tensor(title).cuda()
title = title.unsqueeze(0)
ing = torch.tensor(ingredients).cuda()
ing = ing.unsqueeze(0)
nz = cfg.GAN.Z_DIM
noise = Variable(torch.FloatTensor(1, nz))
fixed_noise = Variable(torch.FloatTensor(1, nz).normal_(0, 1))
def plot_image_with_pil(image_tensor):
    # 将张量转换为PIL图像
    image = image_tensor.cpu().detach()
    image = torch.clamp(image, min=-1, max=1)  # 确保数据在正确的范围内
    image = (image + 1) / 2.0  # 将数据从[-1, 1]转换到[0, 1]
    image = image.squeeze(0).permute(1, 2, 0)  # 调整维度 CxHxW => HxWxC
    image = (image.numpy() * 255).astype('uint8')
    image = Image.fromarray(image)

    # 显示图像
    image.show()

def show_images(images, num_images=1):
    for index in range(num_images):
        image = images[index].cpu().detach().numpy()
        image = np.transpose(image[0], (1, 2, 0))  # CxHxW => HxWxC
        image = (image + 1) / 2.0  # [-1, 1] => [0, 1]
        plt.imshow(image)
        plt.axis('off')
        plt.show()

def plot_images(images, num_images=5):
    plt.figure(figsize=(num_images * 5, 5))


    for index in range(num_images):
        image = images[index].cpu().detach().numpy()
        image = np.transpose(image[0], (1, 2, 0))  # CxHxW => HxWxC
        image = (image + 1) / 2.0  # [-1, 1] => [0, 1]

        plt.subplot(1, num_images, index + 1)
        plt.imshow(image)
        plt.axis('off')

    plt.show()

with torch.no_grad():
    ins_features = get_embedding(instructions, tokenizer, ins_encoder)
    ins_features = linear_ins(ins_features)
    ins_feat = torch.tensor(ins_features).cuda()

    txt_feat = text_encoder([title, ing])
    print(txt_feat.shape)
    print(ins_feat.shape)
    txt_embedding = torch.cat([txt_feat, ins_feat], dim=1)

    print("Generating fake imgs...")
    noise.data.normal_(0, 1)
    imgs = []
    for i in range(5):
        fake_imgs, _, _ = netG(fixed_noise, txt_embedding)
        imgs.append(fake_imgs[-1])

    plot_images(imgs)








