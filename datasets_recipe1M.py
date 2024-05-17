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

from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel

sys.path.append('../')
from word2vec.get_wordvec import load_recipes, get_title_wordvec, get_ingredients_wordvec, get_instructions_wordvec


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None, branch_num=1):
    try:
        img = Image.open(img_path).convert('RGB')
    except:
        return False
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if branch_num == 0:
        ret.append(normalize(img))
    else:
        for i in range(branch_num):
            if i < (branch_num - 1):
                re_img = transforms.Resize(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


def choose_one_image_path(rcp, img_dir):
    part = rcp['partition']
    image_infos = rcp['images']
    if part == 'train':
        # We do only use the first five images per recipe during training
        imgIdx = np.random.choice(range(min(5, len(image_infos))))
    else:
        imgIdx = 0

    # Find images through path
    loader_path = [image_infos[imgIdx]['id'][i] for i in range(4)]  # [4, 1, f, 7]
    loader_path = os.path.join(*loader_path)  # '4\\1\\f\\7'
    if 'plus' in img_dir:
        path = os.path.join(img_dir, loader_path, image_infos[imgIdx]['id'])
    else:
        path = os.path.join(img_dir, part, loader_path, image_infos[imgIdx]['id'])
    return path


def try_get_imgs(rcp, img_dir, imsize, bbox=None, transform=None, normalize=None, branch_num=1):
    part = rcp['partition']
    image_infos = rcp['images']
    if part == 'train':
        # We do only use the first five images per recipe during training
        imgIdx = np.random.choice(range(min(5, len(image_infos))))
    else:
        imgIdx = 0

    # Find images through path
    loader_path = [image_infos[imgIdx]['id'][i] for i in range(4)]  # [4, 1, f, 7]
    loader_path = os.path.join(*loader_path)  # '4\\1\\f\\7'
    if 'plus' in img_dir:
        path = os.path.join(img_dir, loader_path, image_infos[imgIdx]['id'])
    else:
        path = os.path.join(img_dir, part, loader_path, image_infos[imgIdx]['id'])

    img = Image.open(path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(branch_num):
        if i < (branch_num - 1):
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret


def get_embedding(text, tokenizer, text_encoder):
    token = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = text_encoder(**token)
    features = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size768)
    return features


class FoodDataset(data.Dataset):
    def __init__(
            self,
            recipe_file,
            img_dir,
            levels=1,
            word2vec_file='word2vec/word2vec_recipes.bin',
            vocab_ingrs_file='word2vec/list_of_merged_ingredients.txt',
            part='train',
            food_type='salad',
            base_size=64,
            transform=None,
            num_samples=None,
            bert_type='default'):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.imsize = []
        self.levels = levels
        self.recipe_file = recipe_file
        self.img_dir = img_dir
        for _ in range(levels):
            self.imsize.append(base_size)
            base_size = base_size * 2
        
        print('imsize', self.imsize)
        
        self.recipes = load_recipes(recipe_file, part)
        # 料理の種類を指定
        if food_type:
            self.recipes = [x for x in self.recipes if food_type.lower() in x['title'].lower()]
        # 10000枚の画像で実験
        if num_samples > 0:
            N = min(len(self.recipes), num_samples)
#             self.recipes = np.random.choice(self.recipes, N, replace=False)
            self.recipes = self.recipes[:N]

        wv = KeyedVectors.load(word2vec_file, mmap='r')
        # 从 word2vec_file 文件中加载预训练的 word2vec 向量
        # mmap='r' 是一个内存映射选项，允许对存储在磁盘上的大文件进行部分读取，而不是一次性加载整个文件。这提高访问速度。
        w2i = {w: i + 2 for i, w in enumerate(wv.index_to_key)}
        # 将每个单词映射到一个唯一的整数ID。
        # wv.index_to_key 是一个列表，包含 word2vec 模型中所有单词的词汇表。它按照单词在词汇表中的顺序进行排序
        # enumerate(wv.index_to_key) 会返回每个单词及其索引 (i, w)
        # i+2 为每个单词的ID增加了2。这是为了预留出ID 1 和 2 用于特殊的用途
        w2i['<other>'] = 1
        self.w2i = w2i
        self.i2w = {index: word for word, index in w2i.items()}

        with open(vocab_ingrs_file, 'r') as f:
            vocab_ingrs = f.read().strip().split('\n')
            self.ingr2i = {ingr: i for i, ingr in enumerate(vocab_ingrs)}
        if bert_type == 'default':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        elif bert_type == 'distil':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # 创建一个线性层
        self.linear_ing = nn.Linear(768, 128)
        self.linear_ins = nn.Linear(768, 256)  # 384



    def __getitem__(self, index):
        rcp = self.recipes[index]
        # title = rcp['title']
        # words = title.split()
        instructions = rcp['instructions']
        instructions = ' '.join(instructions)

        title, n_words_in_title = get_title_wordvec(rcp, self.w2i)  # np.int [max_len]  text to index
        
        ingredients, n_ings = get_ingredients_wordvec(rcp, self.w2i, self.ingr2i)  # np.int [max_len]
        # txt_w2v = [title, n_words_in_title, ingredients, n_ingrs]
#         print('ing:',rcp['ingredients'])
#         print(ingredients, n_ings)
        with torch.no_grad():
            # ing_features = get_embedding(ingredients, self.tokenizer, self.text_encoder)
            ins_features = get_embedding(instructions, self.tokenizer, self.text_encoder)
            # 使用这个线性层转换特征
            # ing_features = self.linear_ing(ing_features)
            ins_features = self.linear_ins(ins_features)
            # rcp_features = torch.cat((ins_features, ing_features), 1)
        # instructions, n_insts, n_words_each_inst = get_instructions_wordvec(rcp, self.w2i)  # np.int [max_len, max_len]
        # txt = (title, n_words_in_title, ingredients, n_ingrs, instructions, n_insts, n_words_each_inst)
        img_name = choose_one_image_path(rcp, self.img_dir)  
        while get_imgs(img_name, self.imsize, transform=self.transform, normalize=self.norm, branch_num=self.levels) == False:
            img_name = choose_one_image_path(rcp, self.img_dir)
        imgs = get_imgs(img_name, self.imsize, transform=self.transform, normalize=self.norm, branch_num=self.levels)

        all_idx = range(len(self.recipes))
        wrong_idx = np.random.choice(all_idx)
        while wrong_idx == index:
            wrong_idx = np.random.choice(all_idx)
        wrong_img_name = choose_one_image_path(self.recipes[wrong_idx], self.img_dir)
        wrong_imgs = get_imgs(wrong_img_name, self.imsize, transform=self.transform, normalize=self.norm,
                              branch_num=self.levels)

        # return txt, imgs, wrong_imgs, rcp['title']
        return title, ingredients, n_ings, ins_features, imgs, wrong_imgs, rcp['title']


    def __len__(self):
        return len(self.recipes)


# For test
if __name__ == '__main__':
    class Args:
        pass


    args = Args()
    args.base_size = 64
    args.levels = 1
    args.recipe_file = 'D:/CookGAN/data/recipes_withImage.json'
    args.img_dir = 'D:/CookGAN/data/Recipe1M/images'
    args.food_type = 'salad'
    args.batch_size = 32
    args.workers = 4

    imsize = args.base_size * (2 ** (args.levels - 1))
    train_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    print('Prepare the train set...')
    train_set = FoodDataset(
        recipe_file=args.recipe_file,
        img_dir=args.img_dir,
        levels=args.levels,
        part='train',
        food_type=args.food_type,
        base_size=args.base_size,
        transform=train_transform)
    print('Download the train set...')
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        drop_last=False, shuffle=False, num_workers=int(args.workers))
    print('Dataset process is done.')
    print(train_loader)
    for txt, imgs, w_imgs, title in train_loader:
        print('ok')
        print(len(txt))
        for one_txt in txt:
            print(one_txt.shape)

        print(len(imgs))
        for img in imgs:
            print(img.shape)

        print(len(w_imgs))
        for img in w_imgs:
            print(img.shape)

        print(len(title))
        print(title[0])
        input()
