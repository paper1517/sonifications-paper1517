import os
from threading import main_thread
import numpy as np
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
from src.optim.ntxentloss import NTXentLoss
from src.optim.contrastive_accuracy import ContrastiveAccuracy
from src.data.utils import _collate_fn_raw, _collate_fn_raw_multiclass, _collate_fn_contrastive
from torch.utils.data import DataLoader, sampler
from src.data.raw_transforms import get_raw_contrastive_transforms, simple_raw_contrastive_transforms_v3
from src.data.raw_contrastive_dataset import RawContrastiveDataset
from src.utilities.config_parser import parse_config, get_data_info, get_config
from src.models.model_helper import get_feature_extractor
from src.utilities.training_utils import setup_tpu_dataloaders, tpu_optimization_helper
from src.optim.agc import adaptive_clip_grad
from src.models.contrastive_model import Model
import argparse
from src.data.raw_dataset import RawWaveformDataset as SpectrogramDataset
import wandb
import pickle
import copy
from sklearn.metrics import average_precision_score


parser = argparse.ArgumentParser()
parser.description = "Training script for FSD50k baselines"
parser.add_argument("--cfg_file", type=str,
                    help='path to cfg file')
parser.add_argument("--expdir", "-e", type=str,
                    help="directory for logging and checkpointing")
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--cw", type=str, required=False,
                    help="path to serialized torch tensor containing class weights")
parser.add_argument("--resume_from", type=str,
                    help="checkpoint path to continue training from")
parser.add_argument('--mixer_prob', type=float, default=0.75,
                    help="background noise augmentation probability")
parser.add_argument("--fp16", action="store_true",
                    help='flag to train in FP16 mode')
parser.add_argument("--random_clip_size", type=float, default=5)
parser.add_argument("--val_clip_size", type=float, default=5)
parser.add_argument("--use_mixers", action="store_true")
parser.add_argument("--use_mixup", action="store_true")
parser.add_argument("--prefetch_factor", type=int, default=4)
parser.add_argument("--tpus", type=int, default=1)
parser.add_argument("--log_steps", default=10, type=int)
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--high_aug", action="store_true")
parser.add_argument("--wandb_project", type=str, default="pgr-thesis-contrastive")
parser.add_argument("--wandb_group", type=str, default="")
parser.add_argument("--apply_random_gain", action="store_true")
parser.add_argument("--apply_gaussian_noise", action="store_true")
parser.add_argument("--apply_time_masking", action="store_true")
parser.add_argument("--bg_noise_path", default=None, type=str)
parser.add_argument("--do_linear_eval", action="store_true")
parser.add_argument("--wandb_watch_model", action="store_true")
parser.add_argument("--continue_from_ckpt", type=str, default=None)
parser.add_argument("--continue_from_hparams", type=str, default=None)
parser.add_argument("--ntxent_temp", type=float, default=0.1)
parser.add_argument("--random_seed", type=int, default=8881)


ARGS = parser.parse_args()
ARGS.output_directory = os.path.join(ARGS.expdir, "ckpts")
ARGS.log_directory = os.path.join(ARGS.expdir, "logs")


def calculate_mAP(preds, gts, mixup=False):
    preds = torch.cat(preds, 0).numpy()
    gts = torch.cat(gts, 0).numpy()
    if mixup:
        gts[gts >= 0.5] = 1
        gts[gts < 0.5] = 0
    map_value = average_precision_score(gts, preds, average="weighted")
    return map_value


def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer)


def save_checkpoint(model, optimizer, scheduler, epoch, 
                    tr_loss, tr_acc,
                    linear_eval=False, linear_tr_acc=None, linear_val_acc=None):
    archive = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "tr_loss": tr_loss,
        # "val_loss": val_loss,
        "tr_acc": tr_acc,
        # "val_acc": val_acc
    }
    if linear_eval:
        # archive['supervised_tr_acc'] = linear_tr_acc
        archive['supervised_val_acc'] = linear_val_acc

    if linear_eval:
        ckpt_path = os.path.join(ARGS.output_directory,
                                 "epoch={:03d}_tr_loss={:.6f}_tr_acc={:.6f}_supervised_val={:.6f}.pth".format(
                                     epoch, tr_loss, tr_acc, linear_val_acc
                                 ))
    else:
        ckpt_path = os.path.join(ARGS.output_directory,
                                 "epoch={:03d}_tr_loss={:.6f}_tr_acc={:.6f}.pth".format(
                                     epoch, tr_loss, tr_acc,
                                 ))
    xm.save(archive, ckpt_path)
    xm.master_print("Checkpoint written to -> {}".format(ckpt_path))


def load_checkpoint(ckpt_path, model, optimizer, scheduler):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch']


def train(ARGS):
    np.random.seed(ARGS.random_seed)
    torch.manual_seed(ARGS.random_seed)
    cfg = get_config(ARGS.cfg_file)
    mode = cfg['model']['type']
    tpu_world_size = xm.xrt_world_size()
    tpu_local_rank = xm.get_ordinal()
    ac = cfg['audio_config']
    random_clip_size = int(ac['random_clip_size'] * ac['sample_rate'])
    val_clip_size = int(ac['val_clip_size'] * ac['sample_rate'])
    ac['apply_random_gain'] = ARGS.apply_random_gain
    ac['apply_gaussian_noise'] = ARGS.apply_gaussian_noise
    ac['apply_time_masking'] = ARGS.apply_time_masking
    ac['bg_noise_path'] = ARGS.bg_noise_path
    cfg['audio_config'] = ac

    tr_tfs = simple_raw_contrastive_transforms_v3(True, random_clip_size, ac)

    val_tfs = simple_raw_contrastive_transforms_v3(False, val_clip_size, ac)

    # if not ARGS.high_aug:
    #     print("Using simple_raw_contrastive_transforms_v2")

    # else:
    #     tr_tfs = get_raw_contrastive_transforms(random_clip_size, sample_rate=ac['sample_rate'],
    #                                             min_duration=ac['min_duration'],
    #                                             background_noise_path=cfg['data']['background_noise_dir'])
    #     val_tfs = get_raw_contrastive_transforms(val_clip_size, sample_rate=ac['sample_rate'],
    #                                              min_duration=ac['min_duration'],
    #                                              background_noise_path=None)

    assert cfg['model']['type'] == "contrastive"
    train_set = RawContrastiveDataset(cfg['data']['train'], audio_config=ac,
                                      transform=tr_tfs, is_val=False)
    val_set = RawContrastiveDataset(cfg['data']['val'], audio_config=ac,
                                    transform=val_tfs, is_val=True)
    batch_size = cfg['opt']['batch_size']

    device = xm.xla_device()
    model = Model(cfg['model'], ARGS.do_linear_eval).to(device)
    writer = None
    wandb_logger = None
    if xm.is_master_ordinal():
        if not os.path.exists(ARGS.output_directory):
            os.makedirs(ARGS.output_directory)

        if not os.path.exists(ARGS.log_directory):
            os.makedirs(ARGS.log_directory)
        log_name = ARGS.log_directory.split("/")[-2]
        print("RUN NAME:", log_name)
        writer = test_utils.get_summary_writer(ARGS.log_directory)
        if not ARGS.no_wandb:
            wandb_logger = wandb.init(project='{}'.format(ARGS.wandb_project), 
                                      config=cfg, name=log_name,
                                      group=ARGS.wandb_group)
        print(model)
        # save the hyper-parameters to the disk
        with open(os.path.join(ARGS.expdir, "hparams.pickle"), "wb") as handle:
            args_to_save = copy.deepcopy(ARGS)
            args_to_save.cfg = cfg
            pickle.dump(args_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if wandb_logger and ARGS.wandb_watch_model:
        wandb_logger.watch(model, log="all", log_freq=100)
        xm.master_print("Wandb watch enabled")

    train_loader, val_loader = setup_tpu_dataloaders(train_set, val_set,
                                                     tpu_world_size=tpu_world_size, tpu_local_rank=tpu_local_rank,
                                                     batch_size=batch_size, collate_fn=_collate_fn_contrastive,
                                                     num_workers=ARGS.num_workers, multi_tpu_val=True, need_val=ARGS.do_linear_eval)
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    if ARGS.do_linear_eval:
        val_device_loader = pl.MpDeviceLoader(val_loader, device)
    num_steps_per_epoch = len(train_loader)
    optimizer, scheduler, scheduler_name = tpu_optimization_helper(model.parameters(), cfg, ARGS.tpus,
                                                                   reduce_on_plateau_mode="min", 
                                                                   num_tr_steps_per_epoch=num_steps_per_epoch, 
                                                                   num_epochs=ARGS.epochs)
    if ARGS.continue_from_ckpt:
        xm.master_print("Attempting to load checkpoint {}".format(ARGS.continue_from_ckpt))
        epoch = load_checkpoint(ARGS.continue_from_ckpt, model, optimizer, scheduler)
        xm.master_print("Checkpoint loading successful.. Continuing training from Epoch {}".format(epoch))
    else:
        epoch = 1
    if ARGS.do_linear_eval:
        linear_loss_fn = nn.BCEWithLogitsLoss()
    else:
        linear_loss_fn = None
    clip_factor = float(cfg['opt'].get("agc_clip_factor", 0.001))
    loss_fn = NTXentLoss(temperature=ARGS.ntxent_temp)
    ca = ContrastiveAccuracy()
    torch.set_grad_enabled(True)
    # training loop
    min_val_loss = 100.
    agc_clip = bool(cfg['opt'].get("agc_clipping", False))
    for epoch in range(epoch, ARGS.epochs + 1):
        xm.master_print("Epoch {:03d} begin at {}".format(epoch, test_utils.now()))
        tr_step_counter = 0
        model.train()
        tracker = xm.RateTracker()
        tr_contrastive_loss = []
        tr_contrastive_accs = []
        tr_total_loss = []
        if ARGS.do_linear_eval:
            tr_preds = []
            tr_gts = []
            linear_tr_losses = []
            val_preds = []
            val_gts = []
            linear_val_losses = []

        for batch in train_device_loader:
            batch_xi, batch_xj, _, labels = batch
            batch_zi, supervised_zi = model(batch_xi)
            batch_zj, supervised_zj = model(batch_xj)
            # batch_z = torch.cat([batch_zi, batch_zj], dim=0)
            loss = loss_fn(batch_zi, batch_zj)
            con_acc = ca(batch_zi, batch_zj)
            tr_contrastive_loss.append(loss.item())
            if ARGS.do_linear_eval:
                linear_model_targets = torch.cat([labels, labels], dim=0)
                linear_pred = torch.cat([supervised_zi, supervised_zj], dim=0)
                y_pred_sigmoid = torch.sigmoid(linear_pred)
                linear_loss = linear_loss_fn(linear_pred, linear_model_targets)
                linear_tr_losses.append(linear_loss.item())

                loss += linear_loss

            optimizer.zero_grad()
            loss.backward()
            tr_total_loss.append(loss.item())
            if agc_clip:
                xm.master_print("Applying AGC Clipping..")
                adaptive_clip_grad(model.feature_extractor.parameters(), clip_factor=clip_factor)
            xm.optimizer_step(optimizer)
            tracker.add(batch_xi.size(0))
            if tr_step_counter % ARGS.log_steps == 0:
                xm.add_step_closure(
                    _train_update, args=(device, tr_step_counter, loss, tracker, epoch, writer)
                )
            
            tr_contrastive_accs.append(con_acc.item())
            tr_step_counter += 1
            if scheduler_name == "warmupcosine":
                scheduler.step()

        mean_tr_contrastive_loss = np.mean(tr_contrastive_loss)
        mean_tr_contrastive_accs = np.mean(tr_contrastive_accs)
        epoch_tr_contrastive_loss = xm.mesh_reduce("tr_contrastive_loss", mean_tr_contrastive_loss, np.mean)
        epoch_tr_contrastive_acc = xm.mesh_reduce("tr_con_acc", mean_tr_contrastive_accs, np.mean)

        mean_tr_total_loss = np.mean(tr_total_loss)
        epoch_tr_total_loss = xm.mesh_reduce("tr_total_loss", mean_tr_total_loss, np.mean)

        if ARGS.do_linear_eval:
            mean_linear_tr_loss = np.mean(linear_tr_losses)
            epoch_linear_tr_loss = xm.mesh_reduce("linear_tr_losses", mean_linear_tr_loss, np.mean)
            # linear_train_accuracy = calculate_mAP(tr_preds, tr_gts, False)
            # linear_train_accuracy = xm.mesh_reduce("linear_train_accuracy", linear_train_accuracy, np.mean)

        val_step_counter = 0
        model.eval()
        
        val_loss = []
        val_contrastive_accs = []
        if ARGS.do_linear_eval:
            for batch in val_device_loader:
                batch_xi, batch_xj, _, labels = batch
                with torch.no_grad():
                    _, supervised_zi = model(batch_xi)
                    _, supervised_zj = model(batch_xj)
                    linear_model_targets = torch.cat([labels, labels], dim=0)
                    linear_pred = torch.cat([supervised_zi, supervised_zj], dim=0)
                    y_pred_sigmoid = torch.sigmoid(linear_pred)
                    val_preds.append(y_pred_sigmoid.detach().cpu().float())
                    val_gts.append(linear_model_targets.detach().cpu().float())

            # loss = loss_fn(batch_zi, batch_zj)
            # con_acc = ca(batch_zi, batch_zj)
            # val_contrastive_accs.append(con_acc.item())
            # val_loss.append(loss.item())

        if ARGS.do_linear_eval:
            linear_val_accuracy = calculate_mAP(val_preds, val_gts, False)
            linear_val_accuracy = xm.mesh_reduce("linear_val_accuracy", linear_val_accuracy, np.mean)

        xm.master_print('Epoch {:03d} end at {} epoch total loss: {:.6f} \n \t -> epoch_tr_contrastive_loss: {:.6f} | epoch_tr_contrastive_acc: {:.6f}'.format(
            epoch, test_utils.now(), epoch_tr_total_loss, epoch_tr_contrastive_loss, epoch_tr_contrastive_acc))
        if ARGS.do_linear_eval:
            xm.master_print("\t -> epoch tr supervised loss: {:.6f} | Supervised val_map:{:.6f}".format(
                epoch_linear_tr_loss, linear_val_accuracy))
        # min_val_loss = min(min_val_loss, epoch_val_loss)
        dict_to_write = {
            "train/tr_contrasive_loss": epoch_tr_contrastive_loss,
            "train/tr_contrastive_acc": epoch_tr_contrastive_acc,
        }
        if ARGS.do_linear_eval:
            dict_to_write['train/supervised_tr_loss'] = epoch_linear_tr_loss
            # dict_to_write['train/supervised_tr_acc'] = linear_train_accuracy
            dict_to_write['val/supervised_val_acc'] = linear_val_accuracy
        if wandb_logger:
            wandb_logger.log(dict_to_write)
        test_utils.write_to_summary(
            writer,
            epoch,
            dict_to_write=dict_to_write,
            write_xla_metrics=True
        )
        if scheduler_name == "reduce":
            scheduler.step(epoch_tr_contrastive_loss)
        elif scheduler_name == "step":
            scheduler.step()
        if ARGS.do_linear_eval:
            save_checkpoint(model, optimizer, scheduler, 
                            epoch, epoch_tr_total_loss,
                            epoch_tr_contrastive_acc,
                            ARGS.do_linear_eval, linear_val_accuracy)
        else:
            save_checkpoint(model, optimizer, scheduler, 
                            epoch, epoch_tr_total_loss,
                            epoch_tr_contrastive_acc)
        train_loader, val_loader = setup_tpu_dataloaders(train_set, val_set,
                                                     tpu_world_size=tpu_world_size, tpu_local_rank=tpu_local_rank,
                                                     batch_size=batch_size, collate_fn=_collate_fn_contrastive,
                                                     num_workers=ARGS.num_workers, multi_tpu_val=True, need_val=ARGS.do_linear_eval)
        train_device_loader = pl.MpDeviceLoader(train_loader, device)
        if ARGS.do_linear_eval:
            val_device_loader = pl.MpDeviceLoader(val_loader, device)

    test_utils.close_summary_writer(writer)
    xm.master_print("Training done, best val_loss: {}".format(min_val_loss))
    if wandb_logger:
        wandb_logger.finish()
    return min_val_loss


def _mp_fn(index, flags):
    min_val_loss = train(flags)


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(ARGS,), nprocs=ARGS.tpus)
