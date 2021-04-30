import deepsnap
import networkx as nx
import numpy as np
import torch
from graphgym.config import cfg
from graphgym.register import register_loader
from importlib_resources import files


def load_from_numpy(path):
    adj = np.load(path, allow_pickle=True).tolist()

    # TODO: omitting loading of labels right now
    #  see https://github.com/Ivanopolo/modnet/blob/master/evaluate_models.py#L79
    nxG = nx.convert_matrix.from_scipy_sparse_matrix(adj)

    return nxG


def load_modnet_graph(format, _, dataset_dir):
    if cfg.dataset.interpretation != 'single' or cfg.dataset.format != 'modnet':
        return None

    adj_filename = "adj-" + str(cfg.seed) + ".npy"
    labels_filename = "labels-" + str(cfg.seed) + ".npy"

    base_path = files('data').joinpath('modnet-benchmarks').joinpath(
        dataset_dir).joinpath('train')
    adj_path = base_path.joinpath(adj_filename)
    labels_path = base_path.joinpath(labels_filename)

    print("loading graph", adj_path)

    nxG: nx.graph.Graph
    nxG = load_from_numpy(adj_path)

    # add dummy "node feature"
    # TODO GG breaks if no node attributes are set.
    for nodeIx in nxG.nodes:
        onehot = np.zeros(1)
        onehot = torch.from_numpy(onehot).to(torch.float)
        nxG.node[nodeIx]['node_feature'] = onehot

    labels = np.load(labels_path)
    # set partition as graph attribute
    partition = []
    for i in range(labels.max() + 1):
        partition.append(set(np.argwhere(labels == i).flatten()))
    nxG.graph['partition'] = partition

    # TODO GG breaks if no labels are set -- even if labels never really used
    # set membership as node attribute
    for nodeIx in nxG.nodes:
        nxG.node[nodeIx]['node_label'] = torch.tensor(labels[nodeIx]).to(torch.float)

    dsG = deepsnap.graph.Graph(nxG)

    return [dsG]


register_loader('modnet_generated_seed', load_modnet_graph)
