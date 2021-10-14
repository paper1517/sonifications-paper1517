import glob
import tqdm
flds = glob.glob("/home/user/my_disk/experiments_speech_commands_layerwise/*/*/")
flds += glob.glob("/home/user/my_disk/experiments_speech_commands_layerwise_latest/*/*/")
import os
to_keep = []
for fld in tqdm.tqdm(flds):
    ckpt = glob.glob(os.path.join(fld, "ckpts", "epoch={:03d}*".format(50)))
    if len(ckpt) == 0:
        continue
    to_keep.append(ckpt[0])
print(len(to_keep))
all_ckpts = glob.glob("/home/user/my_disk/experiments_speech_commands_layerwise/*/*/ckpts/*")
all_ckpts += glob.glob("/home/user/my_disk/experiments_speech_commands_layerwise_latest/*/*/ckpts/*")
for f in tqdm.tqdm(all_ckpts):
    if f not in to_keep:
        os.remove(f)
