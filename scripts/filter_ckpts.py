import argparse
import os
import tqdm
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str)

def get_epoch(x):
    x = x.split("/")[-1]             
    x = x.split("_")[0].split("=")[-1]
    return int(x)

args = parser.parse_args()
to_keep = [1]+[i*10 for i in range(1,11)]
all_ckpts = glob.glob(os.path.join(args.exp_dir, "ckpts", "*"))

to_keep_ckpts = []
for i in to_keep:
    ckpt = glob.glob(os.path.join(args.exp_dir, "ckpts", "epoch={:03d}*".format(i)))
    print(ckpt)
    if len(ckpt) == 0:
        continue
    to_keep_ckpts.append(ckpt[0])

to_remove = [f for f in all_ckpts if f not in to_keep_ckpts]
for f in tqdm.tqdm(to_remove):
    os.remove(f)
