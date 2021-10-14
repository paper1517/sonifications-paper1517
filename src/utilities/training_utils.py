import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader
import transformers


def setup_tpu_dataloaders(train_set, val_set, tpu_world_size,
                          tpu_local_rank, batch_size, collate_fn=None,
                          num_workers=4, multi_tpu_val=False, need_val=True):
    train_sampler = None
    val_sampler = None
    tr_shuffle = True
    if tpu_world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=tpu_world_size,
            rank=tpu_local_rank,
            shuffle=True)
        tr_shuffle = False
        if multi_tpu_val:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_set,
                num_replicas=tpu_world_size,
                rank=tpu_local_rank,
                shuffle=False
            )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=tr_shuffle,
                              sampler=train_sampler,
                              num_workers=num_workers, collate_fn=collate_fn)
    if need_val:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                sampler=val_sampler,
                                num_workers=num_workers, collate_fn=collate_fn)
    else:
        val_loader = None
    return train_loader, val_loader


def tpu_optimization_helper(params, cfg, num_tpus, reduce_on_plateau_mode='max', 
                            num_tr_steps_per_epoch=None, num_epochs=None):
    optimizer_name = cfg['opt'].get("optimizer", "Adam")
    wd = float(cfg['opt'].get("weight_decay", 0))
    lr = float(cfg['opt'].get("lr", 1e-3))
    effective_batch_size = num_tpus * int(cfg['opt'].get('batch_size', 256))
    # lr = lr * num_tpus
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif optimizer_name == "SGD":
        momentum = float(cfg['opt'].get("momentum", 0.9))
        nesterov = bool(cfg['opt'].get("nesterov", True))
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=wd,
                                    momentum=momentum, nesterov=nesterov)
    else:
        raise ValueError("Unsupported optimizer {}".format(optimizer_name))
    scheduler_name = cfg['opt'].get("scheduler", "reduce")
    if scheduler_name == "reduce":
        patience = int(cfg['opt'].get("patience", 15))
        gamma = float(cfg['opt'].get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, reduce_on_plateau_mode, factor=gamma,
                                                               patience=patience, verbose=True,
                                                               min_lr=1e-6, threshold=5e-3)
    elif scheduler_name == "step":
        step_size = int(cfg['opt'].get("step_size", 30))
        gamma = float(cfg['opt'].get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)    
    elif scheduler_name == "warmupcosine":
        print("Using WarmupCosine Schedule")
        assert num_tr_steps_per_epoch is not None
        assert num_epochs is not None
        total_tr_steps = num_tr_steps_per_epoch * num_epochs
        warmup_steps = num_tr_steps_per_epoch * int(cfg['opt'].get("warmup_epochs", 10))
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_tr_steps)
    else:
        raise ValueError("Unsupported scheduler {}".format(scheduler_name))

    return optimizer, scheduler, scheduler_name
