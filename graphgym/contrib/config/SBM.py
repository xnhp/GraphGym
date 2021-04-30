from graphgym.register import register_config


def set_sbm_config(cfg):
    cfg.dataset.sbm_n = 250  # number of nodes
    cfg.dataset.sbm_m = 25000  # number of edges
    cfg.dataset.sbm_c = 4  # number of communities
    cfg.dataset.sbm_k = 3  # hyperedge degree
    cfg.dataset.sbm_p = 1  # probability of a sampled tuple that is wholly
    # intra-community to be added as an edge to the resulting graph
    cfg.dataset.sbm_q = 0.01  # ...that is inter-community...


register_config('sbm_config', set_sbm_config)
