import os
import copy
import pickle
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
from src.data.utils import _collate_fn_raw, _collate_fn_raw_multiclass
from torch.utils.data import DataLoader, sampler
from src.data.raw_transforms import get_raw_transforms, get_raw_transforms_v2, simple_supervised_transforms
from src.utilities.config_parser import parse_config, get_data_info, get_config
from src.models.model_helper import get_feature_extractor
from src.optim.agc import adaptive_clip_grad
from src.utilities.training_utils import setup_tpu_dataloaders, tpu_optimization_helper
import argparse
from src.data.raw_dataset import RawWaveformDataset as SpectrogramDataset
import wandb
from src.data.mixup import do_mixup, mixup_criterion
from src.models.contrastive_model import get_pretrained_weights_for_transfer
from src.utilities.map import calculate_mAP
from collections import OrderedDict
from src.utilities import interpretability_utils


def filter_feature_layers(layer_index, features_module):
    li = []
    if layer_index < 9:
        final_layer = "mp{}".format(layer_index)
    else:
        final_layer = "act{}".format(layer_index)
    features_dim = None
    for k, v in features_module.named_children():
        li.append((k, v))
        if "conv" in k:
            features_dim = v.out_channels
        if k == final_layer:
            break
    return torch.nn.Sequential(OrderedDict(li)), features_dim


class FinetunedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pretrained_hparams_path = cfg['pretrained_hparams_path']
        pretrained_ckpt_path = cfg['pretrained_ckpt_path']
        num_classes = cfg['num_classes']
        if cfg['is_contrastive_pretrained']:
            module, output_dims = get_pretrained_weights_for_transfer(pretrained_hparams_path,
                                                                             pretrained_ckpt_path)
            features = module.features
        else:
            module, _ = interpretability_utils.get_supervised_pretrained_model(pretrained_hparams_path,
                                                                                        pretrained_ckpt_path)
            features = module.features.features
        self.features, output_dims = filter_feature_layers(cfg['feature_layer_index'], features)
        self.fc = nn.Linear(output_dims, num_classes)

    def forward(self, x):
        # have switch storing turned on in MaxPool layer, can't just use self.features.forward
        # so will loop over the layers

        for name, layer in self.features.named_children():
            if isinstance(layer, nn.MaxPool1d):
                x, indices = layer(x)
            else:
                x = layer(x)
        
        # average them for fc
        x = x.mean(2)
        x = self.fc(x)
        return x


def save_checkpoint(model, optimizer, scheduler, epoch,
                    tr_loss, tr_acc, val_acc):
    archive = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "tr_loss": tr_loss,
        "tr_acc": tr_acc,
        "val_acc": val_acc
    }
    ckpt_path = os.path.join(ARGS.output_directory,
                             "epoch={:03d}_tr_loss={:.6f}_tr_acc={:.6f}_val_acc={:.6f}.pth".format(
                                epoch, tr_loss, tr_acc, val_acc
                             ))
    xm.save(archive, ckpt_path)
    xm.master_print("Checkpoint written to -> {}".format(ckpt_path))


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
parser.add_argument("--use_mixers", action="store_true")
parser.add_argument("--use_mixup", action="store_true")
parser.add_argument("--prefetch_factor", type=int, default=4)
parser.add_argument("--tpus", type=int, default=1)
parser.add_argument("--log_steps", default=10, type=int)
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--high_aug", action="store_true")
parser.add_argument("--wandb_project", type=str, default="pgr-thesis-contrastive-ft")
parser.add_argument("--wandb_group", type=str, default="dataset")
parser.add_argument("--labels_delimiter", type=str, default=",")
parser.add_argument("--fc_only", action="store_true")
parser.add_argument("--wandb_watch_model", action="store_true")
parser.add_argument("--feature_layer_index", type=int, default=1)
parser.add_argument("--is_contrastive_pretrained", action="store_true")
parser.add_argument("--random_seed", type=int, default=8881)


ARGS = parser.parse_args()
ARGS.output_directory = os.path.join(ARGS.expdir, "ckpts")
ARGS.log_directory = os.path.join(ARGS.expdir, "logs")


def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer)


def train(ARGS):
    # cfg = parse_config(ARGS.cfg_file)
    np.random.seed(ARGS.random_seed)
    torch.manual_seed(ARGS.random_seed)
    cfg = get_config(ARGS.cfg_file)
    # data_cfg = get_data_info(cfg['data'])
    # cfg['data'] = data_cfg
    assert cfg['model']['pretrained_hparams_path']
    assert cfg['model']['pretrained_ckpt_path']

    mode = cfg['model']['type']
    tpu_world_size = xm.xrt_world_size()
    tpu_local_rank = xm.get_ordinal()
    ac = cfg['audio_config']
    random_clip_size = int(ac['random_clip_size'] * ac['sample_rate'])
    val_clip_size = int(ac['val_clip_size'] * ac['sample_rate'])
    # random_clip_size = int(ARGS.random_clip_size * cfg['audio_config']['sample_rate'])
    # val_clip_size = int(ARGS.val_clip_size * cfg['audio_config']['sample_rate'])
    if ARGS.high_aug:
        tr_tfs = get_raw_transforms_v2(True, random_clip_size,
                                       sample_rate=ac['sample_rate'])
        val_tfs = get_raw_transforms_v2(False, val_clip_size, center_crop_val=True,
                                        sample_rate=ac['sample_rate'])
    else:
        tr_tfs = simple_supervised_transforms(True, random_clip_size,
                                       sample_rate=ac['sample_rate'])
        val_tfs = simple_supervised_transforms(False, val_clip_size,
                                        sample_rate=ac['sample_rate'])
    train_set = SpectrogramDataset(cfg['data']['train'],
                                   cfg['data']['labels'],
                                   cfg['audio_config'],
                                   mode=mode, augment=True,
                                   mixer=None,  delimiter=ARGS.labels_delimiter,
                                   transform=tr_tfs, is_val=False,
                                   use_tpu=True,
                                   tpu_local_rank=tpu_local_rank,
                                   tpu_world_rank=tpu_world_size)

    val_set = SpectrogramDataset(cfg['data']['val'],
                                 cfg['data']['labels'],
                                 cfg['audio_config'],
                                 mode=mode, augment=False,
                                 mixer=None,  delimiter=ARGS.labels_delimiter,
                                 transform=val_tfs, is_val=True,
                                 use_tpu=True,
                                 tpu_local_rank=tpu_local_rank,
                                 tpu_world_rank=tpu_world_size)

    batch_size = cfg['opt']['batch_size']

    device = xm.xla_device()
    cfg['model']['feature_layer_index'] = ARGS.feature_layer_index
    cfg['model']['is_contrastive_pretrained'] = ARGS.is_contrastive_pretrained
    model = FinetunedModel(cfg['model']).to(device)
    if ARGS.fc_only:
        for param in model.features.parameters():
            param.requires_grad = False
    collate_fn = _collate_fn_raw_multiclass if mode == "multiclass" else _collate_fn_raw
    train_loader, val_loader = setup_tpu_dataloaders(train_set, val_set,
                                                     tpu_world_size=tpu_world_size, tpu_local_rank=tpu_local_rank,
                                                     batch_size=batch_size, collate_fn=collate_fn,
                                                     num_workers=ARGS.num_workers, multi_tpu_val=False)
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    val_device_loader = pl.MpDeviceLoader(val_loader, device)
    num_steps_per_epoch = len(train_loader)
    if ARGS.fc_only:
        optimizer, scheduler, scheduler_name = tpu_optimization_helper(model.fc.parameters(), cfg, ARGS.tpus,
                                                                    reduce_on_plateau_mode="max",
                                                                    num_tr_steps_per_epoch=num_steps_per_epoch,
                                                                    num_epochs=ARGS.epochs)
    else:
        optimizer, scheduler, scheduler_name = tpu_optimization_helper(model.parameters(), cfg, ARGS.tpus,
                                                                    reduce_on_plateau_mode="max",
                                                                    num_tr_steps_per_epoch=num_steps_per_epoch,
                                                                    num_epochs=ARGS.epochs)

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
                                      group="{}".format(ARGS.wandb_group),
                                      config=cfg, name=log_name)
        print(model)
        with open(os.path.join(ARGS.expdir, "hparams.pickle"), "wb") as handle:
            args_to_save = copy.deepcopy(ARGS)
            args_to_save.cfg = cfg
            pickle.dump(args_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    clip_factor = float(cfg['opt'].get("agc_clip_factor", 0.001))
    if mode == "multiclass":
        loss_fn = nn.CrossEntropyLoss()
    elif mode == "multilabel":
        loss_fn = nn.BCEWithLogitsLoss()
    if wandb_logger and ARGS.wandb_watch_model:
        wandb_logger.watch(model, log="all", log_freq=100)
    mixup_enabled = cfg["audio_config"].get("mixup", False) #and mode == "multilabel"
    if mixup_enabled:
        xm.master_print("Attention: Will use mixup while training..")

    torch.set_grad_enabled(True)

    # if wandb_logger:
    #     wandb_logger.watch(model)
    agc_clip = bool(cfg['opt'].get("agc_clipping", False))
    accuracy, max_accuracy = 0.0, 0.0
    for epoch in range(1, ARGS.epochs + 1):
        xm.master_print("Epoch {:03d} train begin {}".format(epoch, test_utils.now()))
        tr_step_counter = 0
        model.train()
        tracker = xm.RateTracker()
        tr_loss = []
        tr_correct = 0
        tr_total_samples = 0

        tr_preds = []
        tr_gts = []

        for batch in train_device_loader:
            x, _, y = batch
            if mixup_enabled:
                if mode == "multilabel":
                    x, y, _, _ = do_mixup(x, y, mode=mode)
                elif mode == "multiclass":
                    x, y_a, y_b, lam = do_mixup(x, y, mode=mode)
            pred = model(x)
            if mode == "multiclass":
                pred_labels = pred.max(1, keepdim=True)[1]
                tr_correct += pred_labels.eq(y.view_as(pred_labels)).sum()
                tr_total_samples += x.size(0)
                if mixup_enabled:
                   loss = mixup_criterion(loss_fn, pred, y_a, y_b, lam)
                else:
                   loss = loss_fn(pred, y)
            else:
                y_pred_sigmoid = torch.sigmoid(pred)
                tr_preds.append(y_pred_sigmoid.detach().cpu().float())
                tr_gts.append(y.detach().cpu().float())
                loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            if agc_clip:
                adaptive_clip_grad(model.features.parameters(), clip_factor=clip_factor)
            xm.optimizer_step(optimizer)
            tracker.add(batch_size)
            if tr_step_counter % ARGS.log_steps == 0:
                xm.add_step_closure(
                    _train_update, args=(device, tr_step_counter, loss, tracker, epoch, writer)
                )
                # if wandb_logger:
                #     wandb_logger.log({"batch_tr_loss": loss})
            tr_loss.append(loss.item())
            tr_step_counter += 1
            if scheduler_name == "warmupcosine":
                scheduler.step()
        mean_tr_loss = np.mean(tr_loss)
        epoch_tr_loss = xm.mesh_reduce("tr_loss", mean_tr_loss, np.mean)
        if mode == "multiclass":
            tr_acc = tr_correct.item() / tr_total_samples
        else:
            # calculate mAP
            tr_acc = calculate_mAP(tr_preds, tr_gts, mixup_enabled, mode="weighted")

        tr_acc = xm.mesh_reduce("train_accuracy", tr_acc, np.mean)
        xm.master_print('Epoch {} train end {} | Mean Loss: {} | Mean Acc:{}'.format(epoch,
                        test_utils.now(), epoch_tr_loss, tr_acc))
        val_step_counter = 0
        model.eval()
        total_samples = 0
        correct = 0
        del tr_gts, tr_preds
        if xm.is_master_ordinal():
            val_preds = []
            val_gts = []
            xm.master_print("Validating..")
            for batch in val_device_loader:
                x, _, y = batch
                with torch.no_grad():
                    pred = model(x)
                    # xm.master_print("pred.shape:", pred.shape)
                if mode == "multiclass":
                    pred = pred.max(1, keepdim=True)[1]
                    correct += pred.eq(y.view_as(pred)).sum()
                    total_samples += x.size()[0]
                else:
                    y_pred_sigmoid = torch.sigmoid(pred)
                    val_preds.append(y_pred_sigmoid.detach().cpu().float())
                    val_gts.append(y.detach().cpu().float())
            if mode == "multiclass":
                accuracy = correct.item() / total_samples
            else:
                # xm.master_print("calculating map")
                accuracy = calculate_mAP(val_preds, val_gts)
                # xm.master_print("mAP calculated")
                # val_preds = torch.cat(val_preds, 0)
                # val_gts = torch.cat(val_gts, 0)
                # all_val_preds = xm.mesh_reduce("all_val_preds", val_preds, torch.cat)
                # xm.master_print("after all reduce, preds shape:", all_val_preds.shape)
            # accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)

            xm.master_print('Epoch {} test end {}, Accuracy={:.4f}'.format(
            epoch, test_utils.now(), accuracy))
            max_accuracy = max(accuracy, max_accuracy)
            dict_to_write = {
                "tr_loss": epoch_tr_loss,
                "tr_acc": tr_acc,
                # "val_loss": epoch_val_loss,
                "val_acc": accuracy
            }
            del val_gts, val_preds
            if wandb_logger:
                wandb_logger.log(dict_to_write)
            test_utils.write_to_summary(
                writer,
                epoch,
                dict_to_write=dict_to_write,
                write_xla_metrics=True)
        save_checkpoint(model, optimizer, scheduler, epoch, epoch_tr_loss, tr_acc, accuracy)
        if scheduler_name == "reduce":
            scheduler.step(tr_acc)
        else:
            scheduler.step()

    test_utils.close_summary_writer(writer)
    xm.master_print("Training done, best acc: {}".format(max_accuracy))
    if wandb_logger:
        wandb_logger.finish()
    return max_accuracy


def _mp_fn(index, flags):
    # torch.set_default_tensor_type("torch.FloatTensor")
    acc = train(flags)


if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=(ARGS,), nprocs=ARGS.tpus)
