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
files = glob.glob("{}/*.{}".format(args.src_dir, args.src_ext))

tgt_dir = args.tgt_dir
if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

SAMPLE_RATE = args.sr
lf = len(files)


def process_idx(idx):
    f = files[idx]
    fname = f.split("/")[-1]
    fname = fname.replace(".{}".format(args.src_ext), ".{}".format(args.tgt_ext))
    tgt_path = os.path.join(tgt_dir, fname)

    txt_p = f.replace(".{}".format(args.src_ext), ".txt")
    tgt_txt_p = tgt_path.replace(".{}".format(args.tgt_ext), ".txt")

    # print(tgt_path)
    command = "ffmpeg -loglevel 0 -nostats -i '{}' -ac 1 -ar {} '{}' -y".format(f, SAMPLE_RATE, tgt_path)
    # print(command)
    sp.call(command, shell=True)
    if os.path.exists(tgt_txt_p):
        sp.call("cp '{}' '{}'".format(txt_p, tgt_txt_p), shell=True)
    if idx % 500 == 0:
        print("Done: {:05d}/{}".format(idx, lf))


if __name__ == '__main__':
    pool = Pool(6)
    o = pool.map_async(process_idx, range(lf))
    res = o.get()
    pool.close()
    pool.join()
