import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time

import wandb

from datasets_recipe1M import FoodDataset

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))

sys.path.append(dir_path)

from miscc.config import cfg, cfg_from_file

CLASS_DIC = {}

# Check if CUDA is available now
root = 'D:/FoodGAN'
os.environ["WANDB__SERVICE_WAIT"] = "300"


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN Network')
    # プログラム実行時にコマンドラインで引数を受け取る処理を簡単に実装
    parser.add_argument('--project_name', type=str, default='FoodGAN')
    # 'FoodGAN-branch1-OSGAN'

    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)  # f'{root}/code/miscc/train_attn.yml'
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')

    parser.add_argument('--data_dir', dest='data_dir', type=str, default='D:/CookGAN/data/Recipe1M')
    # parser.add_argument('--data_dir', dest='data_dir', type=str, default='D:/StackGAN-v2/data/birds')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--checkpoint_efficient_net', type=str, default="efficientnet_lite1.pth", metavar='Path',
                        help='Path for EfficientNet checkpoint (default: efficientnet_lite1.pth)')
    parser.add_argument('--recipe_file', type=str, default=f'D:/CookGAN/data/Recipe1M/recipes_withImage.json')
    parser.add_argument('--img_dir', type=str, default='D:/CookGAN/data/Recipe1M/images')
    parser.add_argument('--word2vec_file', type=str, default=f'{root}/word2vec/word2vec_recipes.bin')
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument('--trainer_type', type=str, default='n')
    parser.add_argument('--load_models', type=int, default=0)
    parser.add_argument('--resume', type=int, default=0)  # wandb resume
    parser.add_argument('--memo', type=str, default='steak')
    # 'onestage'
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA使用可能")
    else:
        device = torch.device("cpu")
        print("CUDA使用不可")
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id == '-1':
        print(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False
    if args.load_models == 0:
        cfg.TRAIN.PATH = ''
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    # 現在日時のdatetimeオブジェクトを取得
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    if args.num_samples == None:
        num = 'None'
    else:
        num = str(args.num_samples)
    if cfg.FOOD_TYPE == None:
        food = ''
    else:
        food = cfg.FOOD_TYPE

    output_dir = '../output/%s_%s_%s_%s_%s' % \
                 (timestamp, str(cfg.TREE.BRANCH_NUM), food, num, args.memo)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        if cfg.DATASET_NAME == 'birds':
            bshuffle = False
            split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        # 複数の Transform を連続して行う Transform を作成
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    dataset = FoodDataset(
        recipe_file=args.recipe_file,
        img_dir=args.img_dir,
        levels=cfg.TREE.BRANCH_NUM,
        part='train',
        food_type=cfg.FOOD_TYPE,
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform,
        num_samples=args.num_samples,
        bert_type='distil')
    i2w = dataset.i2w
#     if cfg.DEBUG == 1:
#         # 打印前10个键值对
#         print('i2w:')
#         for i, (key, value) in enumerate(i2w.items()):
#             print(key, value)
#             if i >= 9:  # 只打印前10个
#                 break

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))
    print('train data info:', len(dataset), len(dataloader))

    # Define models and go to train/evaluate
    if args.trainer_type == 'n':
        from trainer_food import condGANTrainer as trainer
    elif args.trainer_type == 'w':
        from trainer_food import condWGANTrainer as trainer
    elif args.trainer_type == 'onestage':
        from trainer_OSGAN import condGANTrainer as trainer
    elif args.trainer_type == 'attn':
        from trainer_attn import condGANTrainer as trainer
    algo = trainer(output_dir, dataloader, imsize, args.checkpoint_efficient_net, i2w)

    if cfg.DEBUG == 0:
        # project_name = "FoodGAN-A100"
        project_name = args.project_name
        name = "weiyixia031"
        if args.resume == 1:
            wandb.init(project=project_name, entity=name, dir='wandb', config=args, id=cfg.WANDB_ID, resume="must")
        else:
            wandb.init(project=project_name, entity=name, dir='wandb', config=args, name=cfg.WANDB_NAME)

        wandb.config.update(args)

    # time
    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate(split_dir)
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
