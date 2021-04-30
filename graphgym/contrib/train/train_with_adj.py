import logging
import time

import torch
from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.register import register_train
from graphgym.utils.epoch import is_ckpt_epoch


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss_ret = compute_loss(pred, true, batch)
        if len(loss_ret) == 2:
            total_loss, pred_score = loss_ret
            loss_main = torch.tensor(0)
            loss_reg = torch.tensor(0)
        else:
            total_loss, pred_score, loss_main, loss_reg = loss_ret
        total_loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=total_loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            loss_main=loss_main.item(),
                            loss_reg=loss_reg.item())
        time_start = time.time()
    scheduler.step()


def eval_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss_ret = compute_loss(pred, true, batch)
        if len(loss_ret) == 2:  # todo duplicate code: unpacking of loss values
            total_loss, pred_score = loss_ret
            loss_main = None
            loss_reg = None
        else:
            total_loss, pred_score, loss_main, loss_reg = loss_ret
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=total_loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            loss_main=loss_main.item(),
                            loss_reg=loss_reg.item())
        time_start = time.time()


def train_with_adj(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        # train and evaluate on train split
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch, loaders[0])
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


register_train('train_with_adj', train_with_adj)
