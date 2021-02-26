import os
import random
import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Generate smaller NAS-Bench files')
parser.add_argument('--api_loc', default='/disk/scratch_ssd/nasbench201/NASBench_v1_1.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='bench', type=str, help='folder to save results')
parser.add_argument('--arch_start', default=0, type=int)
parser.add_argument('--arch_end', default=15625, type=int)
parser.add_argument('--seed', default=42, type=int)

args = parser.parse_args()

import torch
import torch.nn as nn
from nas_201_api import NASBench201API as API

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

ARCH_START, ARCH_END = args.arch_start, args.arch_end

api = API(args.api_loc)
seed = 1

df = []

for arch in range(ARCH_START, ARCH_END):
    info = api.query_by_index(arch)
    cellstr = info.arch_str
    
    cifar10_val = info.get_metrics('cifar10-valid', 'x-valid')['accuracy'] #validation when training on the training split of a train/val/test split

    ### CHOOSE WHICH CIFAR-10 TEST ACC YOU WANT
    cifar10_test = info.get_metrics('cifar10-valid', 'ori-test')['accuracy'] # test accuracy when trained on the training split of train/val/test split
    cifar10_test = info.get_metrics('cifar10', 'ori-test')['accuracy'] # test accuracy when trained on training+val on train/val/test split

    ### CIFAR-100 
    cifar100_test = info.get_metrics('cifar100','x-test')['accuracy']
    cifar100_val  = info.get_metrics('cifar100','x-valid')['accuracy']

    ## ImageNet-16-120
    imagenet_test = info.get_metrics('ImageNet16-120', 'x-test')['accuracy']
    imagenet_val  = info.get_metrics('ImageNet16-120', 'x-valid')['accuracy'] 

    df.append([arch, cellstr, cifar10_test, cifar10_val, cifar100_test, cifar100_val, imagenet_test, imagenet_val])

df = pd.DataFrame(df, columns=['arch','cellstr','cifar10-test','cifar10-val', 'cifar100-test', 'cifar100-val', 'imagenet-test', 'imagenet-val'])

df.to_pickle('mini-bench-arch-cell-accs.pd')
