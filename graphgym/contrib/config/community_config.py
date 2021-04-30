from graphgym.register import register_config


def set_cfg_community(cfg):
    # required to set default value in order to register the option.
    cfg.dataset.num_communities = 2

    # behaves the same as the --repeat argument to main.py. If both are set, the larger
    # value will be used for the effective number of repeats.
    cfg.dataset.repeat = 1

    # weight hyperparameter for the collapse regularisation term in the loss
    cfg.model.collapse_lam = 0.1


register_config('community_config', set_cfg_community)
