import deepsnap
import numpy as np
import torch
from contrib.loader.util import _make_dataset
from deprecated import deprecated
from graphgym.config import cfg
from graphgym.register import register_loader
from importlib_resources import files
from pytictoc import TicToc

from data.biomodels import Network


@deprecated('needs updating, see bipartite case')
def sbml_single_hyper(format, name, dataset_dir):
    # GG simply uses the first registered loader that does not return None so we
    # need to make this check in each loader.
    if cfg.dataset.interpretation != 'hyper' or cfg.dataset.format != 'SBML':
        return None
    # TODO use dataset_dir?
    # TODO: how often isthis method called?
    # TODO: profile how long this conversion takes for large networks
    # right now, this goes sbml -> igraph graph -> pytorch tensor -> dsG/networkx graph
    # for alternative method, see loader/SBM.py
    nw = Network.from_sbml(name)
    pyg_data = nw.to_torch_data_onehot_hyper()  # NOTE incidence edge indices
    return _make_dataset(nw, pyg_data, nw.get_num_hypernodes())


register_loader('sbml_single_hyper', sbml_single_hyper)


def sbml_single_bipartite(format, name, dataset_dir):
    if cfg.dataset.interpretation != 'bipartite' or cfg.dataset.format != 'SBML':
        return None
    nw = Network.from_sbml(name)

    if cfg.dataset.max_node_degree is not None:
        nw = nw.limit_node_degrees(cfg.dataset.max_node_degree)

    if cfg.dataset.max_edge_degree is not None:
        nw = nw.limit_edge_degrees(cfg.dataset.max_edge_degree)

    if cfg.dataset.limit_to_largest_component is True:
        nw = nw.limit_to_largest_component()

    nxG = nw.to_networkx()
    # add dummy "node feature"
    # TODO GG breaks if node_feature is not set
    # TODO GG breaks if no node_label is set
    for nodeIx in nxG.nodes:
        feat = np.zeros(1)
        feat = torch.from_numpy(feat).to(torch.float)
        nxG.node[nodeIx]['node_feature'] = feat
        nxG.node[nodeIx]['node_label'] = 0

    dsG = deepsnap.graph.Graph(nxG)

    return [dsG]


register_loader('sbml_single_bipartite', sbml_single_bipartite)


def rxn_csv(format, name, dataset_dir):
    """
    Load SBML, interpret as reaction graph, set features from csv.
    :param format:
    :param name:
    :param dataset_dir:
    :return:
    """
    if cfg.dataset.interpretation != 'reaction_graph' or cfg.dataset.format != 'SBML' \
            or cfg.dataset.node_attr_file is None:
        return None

    nw = Network.from_sbml(name)
    t = TicToc()
    t.tic()
    if cfg.dataset.max_node_degree is not None:
        nw = nw.limit_node_degrees(cfg.dataset.max_node_degree)
    if cfg.dataset.max_edge_degree is not None:
        nw = nw.limit_edge_degrees(cfg.dataset.max_edge_degree)
    nw = nw.limit_to_largest_component()  # returns igraph.Graph instead of
    # biomodels.Network
    nw = nw.bipartite_projection(which=Network.hyperedge_t)

    import csv
    csv_path = files('data').joinpath(cfg.dataset.node_attr_file)
    attrs: dict
    attrs = {}
    with open(csv_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        # number of features is number of columns minus one for rxn id
        for row in csvReader:
            floats = [float(v) for v in row[1:len(row)]]
            # row[0] is rxn id, all others are considered features
            attrs[row[0]] = floats

    for node in nw.vs:
        node['node_label'] = 0  # TODO GG breaks if no node_label is set
        if node['name'] in attrs:
            node['has_feature'] = True
            # for comparison experiments, consider the same subgraph but without node
            # features
            if cfg.dataset.use_node_feature is False:
                node['node_feature'] = torch.tensor([0]).to(torch.float)
            else:
                node['node_feature'] = torch.tensor(attrs[node['name']]).to(torch.float)

    # consider the induced subgraph of those reactions with non-zero features
    # TODO select only nodes which have features set,
    #   or simply save their ids in the loop above
    nw_sub = nw.induced_subgraph(nw.vs.select(has_feature=True))  # list of node ids
    # nw_sub = nw

    dsG = deepsnap.graph.Graph(nw_sub.to_networkx())
    return [dsG]


register_loader('rxn_csv', rxn_csv)


def mtb_csv(format, name, dataset_dir):
    """
    Load SBML, interpret as reaction graph, set features from csv.
    :param format:
    :param name:
    :param dataset_dir:
    :return:
    """
    if cfg.dataset.interpretation != 'metab_graph' or cfg.dataset.format != 'SBML' \
            or cfg.dataset.node_attr_file is None:
        return None

    nw = Network.from_sbml(name)
    t = TicToc()
    t.tic()
    if cfg.dataset.max_node_degree is not None:
        nw = nw.limit_node_degrees(cfg.dataset.max_node_degree)
    if cfg.dataset.max_edge_degree is not None:
        nw = nw.limit_edge_degrees(cfg.dataset.max_edge_degree)
    nw = nw.to_mtb_graph()  # returns an igraph.Graph instead of a
    # biomodels.Network

    import csv
    csv_path = files('data').joinpath(cfg.dataset.node_attr_file)
    attrs: dict
    attrs = {}
    with open(csv_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        # number of features is number of columns minus one for rxn id
        for row in csvReader:
            floats = [float(v) for v in row[1:len(row)]]
            # row[0] is rxn id, all others are considered features
            attrs[row[0]] = floats

    for node in nw.vs:
        node['node_label'] = 0  # TODO GG breaks if no node_label is set
        if node['name'] in attrs:
            node['has_feature'] = True
            # for comparison experiments, consider the same subgraph but without node
            # features
            if cfg.dataset.use_node_feature is False:
                node['node_feature'] = torch.tensor([0]).to(torch.float)
            else:
                node['node_feature'] = torch.tensor(attrs[node['name']]).to(torch.float)

    # consider the induced subgraph of those reactions with non-zero features
    # TODO select only nodes which have features set,
    #   or simply save their ids in the loop above
    # nw_sub = nw.induced_subgraph(nw.vs.select(has_feature=True))  # list of node ids
    nw_sub = nw

    dsG = deepsnap.graph.Graph(nw_sub.to_networkx())
    return [dsG]


register_loader('mtb_csv', mtb_csv)
