import subprocess as sp
import os
import glob
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str)
parser.add_argument("--tgt_dir", type=str)
parser.add_argument("--sr", type=int, default=8000)
parser.add_argument("--src_ext", type=str, default="flac")
parser.add_argument("--tgt_ext", type=str, default="flac")

args = parser.parse_args()
files = glob.glob("{}/*/*.{}".format(args.src_dir, args.src_ext))

tgt_dir = args.tgt_dir
if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

SAMPLE_RATE = args.sr
lf = len(files)


def process_idx(idx):
    f = files[idx]
    # print(f)
    splitted = f.split("/")
    # print("splitted", splitted)
    fname = splitted[-1]
    fname = fname.replace(".{}".format(args.src_ext), ".{}".format(args.tgt_ext))
    # print("fname:", fname)
    tgt_sub_dir = os.path.join(tgt_dir, splitted[-2])
    # print("tgt_sub_dir", tgt_sub_dir)
    tgt_path = os.path.join(tgt_sub_dir, fname)
    # print("tgt_path", tgt_path)
    if os.path.exists(tgt_path):
        return
    
    if not os.path.exists(tgt_sub_dir):
        os.makedirs(tgt_sub_dir)
    command = "ffmpeg -loglevel 0 -nostats -i '{}' -ac 1 -ar {} '{}' -y".format(f, SAMPLE_RATE, tgt_path)
    sp.call(command, shell=True)
    if idx % 500 == 0:
        print("Done: {:05d}/{}".format(idx, lf))


if __name__ == '__main__':
    pool = Pool(40)
    o = pool.map_async(process_idx, range(lf))
    res = o.get()
    pool.close()
    pool.join()