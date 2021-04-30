from graphgym.register import register_config


def set_sbml_config(cfg):
    # required to set default value in order to register the option.
    # interpret the given network as a hypergraph, bipartite graph, ...
    cfg.dataset.interpretation = 'bipartite'

    cfg.dataset.max_node_degree = None
    cfg.dataset.max_edge_degree = None
    cfg.dataset.limit_to_largest_component = False

    # additional csv file to load node attributes from
    cfg.dataset.node_attr_file = None

    # some loaders will not set node features if set to False
    cfg.dataset.use_node_feature = True


register_config('sbml_config', set_sbml_config)
