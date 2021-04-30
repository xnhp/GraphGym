import deepsnap
from graphgym.config import cfg
from graphgym.register import register_loader
from importlib_resources import files


def SBM_single_bipartite(format, name, dataset_dir):
    """
    Not to be confused with sbml_single_bipartite
    TODO: use dataset.Generator instead?
    """
    if cfg.dataset.interpretation != 'bipartite' or cfg.dataset.format != 'SBM':
        return None

    from util.graph_generators import get_kHSBM_graph
    cd = cfg.dataset
    memberships, nw = get_kHSBM_graph(cd.sbm_n, cd.sbm_m, cd.sbm_k, cd.sbm_p,
                                      cd.sbm_q, cd.sbm_c)
    nxG = nw.to_networkx()
    dsG = deepsnap.graph.Graph(nxG)

    return [dsG]


register_loader('SBM_single_bipartite', SBM_single_bipartite)


def SBM_single_from_generated(format, name, dataset_dir):
    if cfg.dataset.interpretation != 'bipartite' or cfg.dataset.format != 'SBM_gen':
        return None

    # todo generate if not found; coalesce this and `generate_nx_to_file`

    import loader
    dir = files('data').joinpath(dataset_dir)
    dataset = loader.load_nx(name, dir)

    return dataset


register_loader('SBM_single_bip_generated', SBM_single_from_generated)


def SBM_single_from_generated_seed(format, name, dataset_dir):
    if cfg.dataset.interpretation != 'bipartite' or cfg.dataset.format != 'SBM_gen_rpt':
        return None

    # cfg.seed will start at 1 and increase by 1 with every repeat as specified by
    # `--repeat k` as CLI argument to main.py.
    # we use this to pick the k-th pre-generated graph
    import loader
    dir = files('data').joinpath(dataset_dir)
    name += '-' + str(cfg.seed)
    print('loading pre-generated graph ', name)
    dataset = loader.load_nx(name, dir)

    return dataset


register_loader('SBM_single_bip_generated_seed', SBM_single_from_generated_seed)
