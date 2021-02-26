import time
import random 
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Test minibench')
parser.add_argument('--api_loc', default='/disk/scratch_ssd/nasbench201/NASBench_v1_1.pth',
                    type=str, help='path to API')
args = parser.parse_args() 

random.seed(1)

t1 = time.time()
minibench = pd.read_pickle('mini-bench-arch-cell-accs.pd')
t2 = time.time()

print(f"Loading minibench took {t2-t1:.4f} seconds.")

## get 100 random accs 
n_samples = 100
arch_ids = random.sample(range(0, 15625), n_samples)
accs_in_minibench = minibench.iloc[arch_ids]['cifar10-val']

## now get NAS-Bench-201
from nas_201_api import NASBench201API as API

t1 = time.time()
api = API(args.api_loc)
t2 = time.time()

print(f"Loading NAS-Bench-201 took {t2-t1:.4f} seconds.")

arch_acc_equal_to_minibench = []
for arch_id in arch_ids:
    info = api.query_by_index(arch_id)
    cifar10_val = info.get_metrics('cifar10-valid','x-valid')['accuracy']
    arch_acc_equal_to_minibench.append(cifar10_val == accs_in_minibench[arch_id])

if all(arch_acc_equal_to_minibench):
    print("All tests passed.")
else:
    print("!!WARNING!! Tests failed.")
