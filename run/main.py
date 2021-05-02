import logging
import random

import numpy as np
import torch
from graphgym.cmd_args import parse_args
from graphgym.config import (cfg, assert_cfg, dump_cfg,
                             update_out_dir, get_parent_dir)
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import setup_printing, create_logger
from graphgym.model_builder import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device
# NOTE: this import is needed to properly register modules, even though
# IDEs may mark it as unused
from graphgym.contrib.train import *
from graphgym.register import train_dict

if __name__ == '__main__':

    import os

    print(os.environ['PYTHONPATH'])

    # Load cmd line args
    args = parse_args()

    repeats = args.repeat if args.repeat > cfg.dataset.repeat else cfg.dataset.repeat

    # Repeat for different random seeds
    for i in range(args.repeat):
        # Load config file
        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_list(args.opts)
        assert_cfg(cfg)
        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)
        out_dir_parent = cfg.out_dir
        cfg.seed = i + 1
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        update_out_dir(out_dir_parent, args.cfg_file)
        dump_cfg(cfg)
        setup_printing()
        auto_select_device()
        print("using device " + str(cfg.device))
        # Set learning environment
        datasets = create_dataset()
        # create a loader for train split and for any other defined splits
        loaders = create_loader(datasets)
        # create a logger for each loader, i.e. report metrics on train/test/val splits
        meters = create_logger(datasets, loaders)
        # todo: for unsupervised case, specify dim_out explicitly since we do not have
        #   labels to infer shape from. Do this via config.
        if cfg.dataset.task_type == 'community':
            # in unsupervised case, need to specify output dimensionality explicitly
            # since we do not have labels to infer from
            model = create_model(datasets, dim_out=cfg.dataset.num_communities)
        else:
            model = create_model(datasets)
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: {}'.format(cfg.params))
        # Start training
        if cfg.train.mode == 'standard':
            train(meters, loaders, model, optimizer, scheduler)
        else:
            # NOTE: the import "from graphgym.contrib.train import *" is needed
            # to properly import train loop implementations; although an IDE
            # may flag this import as not required
            train_dict[cfg.train.mode](
                meters, loaders, model, optimizer, scheduler)
    # Aggregate results from different seeds
    agg_runs(get_parent_dir(out_dir_parent, args.cfg_file), cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, '{}_done'.format(args.cfg_file))
