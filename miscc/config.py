from __future__ import division
from __future__ import print_function
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C
__C.DEBUG = 0
__C.DATASET_NAME = 'Recipe1M'
__C.FOOD_TYPE = 'steak'
__C.LABELS = 'R-smooth'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''

__C.GPU_ID = '0'
__C.CUDA = True

__C.WORKERS = 2
__C.WANDB_NAME = 'steakall'
__C.WANDB_ID = 'boij9ygv'

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3  # 3
__C.TREE.BASE_SIZE = 64

# Test Options
__C.TEST = edict()
__C.TEST.B_EXAMPLE = True
__C.TEST.SAMPLE_NUM = 30000

# Training Options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 16  # 16
__C.TRAIN.VIS_COUNT = 64  # ?
__C.TRAIN.MAX_EPOCH = 600  # 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4  # 識別器の学習率
__C.TRAIN.GENERATOR_LR = 2e-4  # 生成器の学習率
__C.TRAIN.FLAG = True
__C.TRAIN.PATH = '../output/3_pasta_0_n/Model' # 'models/saladall_1bcr5_t65/'  # '../output/256_salad_10000_160/Model/' # '../output/20231018_005346_3_cake_5000_nocolor_10/Model/' # '../output/256_salad_10000_80_0/Model/'
__C.TRAIN.NET_G = 'netG_60.pth'  # 'netG_180.pth'
__C.TRAIN.NET_D = 'netD'  # 'netD'
__C.TRAIN.TYPE = ''  # 'wgan'
__C.TRAIN.OPT = 'bcr0'  # aug'
__C.TRAIN.DA = 'zt'  # aug'

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 1.0  # 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 1.0  # 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 1.0  # 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 0.0
__C.TRAIN.COEFF.UNCOND_LOSS = 0.0
__C.TRAIN.COEFF.COLOR_LOSS = 0.0
__C.TRAIN.COEFF.CR = 10.0

# Modal Options
__C.GAN = edict()

__C.GAN.EMBEDDING_DIM = 128  # 128
__C.GAN.DF_DIM = 64  # 識別器
__C.GAN.GF_DIM = 64  # 64  # 生成器
__C.GAN.Z_DIM = 100  # Noise
__C.GAN.CONDITION_DIM = 128
__C.GAN.NETWORK_TYPE = 'default'
__C.GAN.R_NUM = 2
__C.GAN.B_CONDITION = True

__C.TEXT = edict()
__C.TEXT.EMBEDDING_DIM = 512  # 256  # テキスト埋め込み次元 1024??
__C.TEXT.WORDS_DIM = 128
__C.TEXT.WORDS_NUM = 18


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return
    for k, v in a.items():  # replace the a.iteritems to the a.items
        # a must specify keys that are in b
        if k not in b:  # not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
            else:
                b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r', encoding="utf-8") as f:
        yaml_cfg = edict(yaml.safe_load(f))  # change f to a dictionary

    _merge_a_into_b(yaml_cfg, __C)
