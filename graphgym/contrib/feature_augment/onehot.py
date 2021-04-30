import networkx as nx
import numpy as np

from graphgym.register import register_feature_augment


def node_onehot_full_fun(graph, **kwargs):
    """
    :param graph: deepsnap graph. graph.G is networkx
    :param kwargs: required, in case additional kwargs are provided
    :return: List of node feature values, length equals number of nodes
    Note: these returned values are later processed and treated as node features
    as specified in "cfg.dataset.augment_feature_repr"
    """
    nxG: nx.Graph
    nxG = graph.G
    features = []
    for nodeIx in nxG.nodes:
        onehot = np.zeros(nxG.number_of_nodes())
        onehot[nodeIx] = 1
        features.append(list(onehot))

    return features


register_feature_augment('node_onehot_full', node_onehot_full_fun)
