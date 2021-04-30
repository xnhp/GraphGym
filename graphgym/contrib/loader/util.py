import deepsnap
import torch
from deprecated import deprecated


@deprecated()
def _make_dataset(nw, pyg_data, num_nodes):
    # labels are not meaningful here for unsupervised case but GraphGym expects them
    pyg_data.y = torch.zeros(num_nodes)
    # TODO does (almost certainly) not work with inc
    dsG = deepsnap.graph.Graph.pyg_to_graph(pyg_data)
    # this would be the more reasonable way, but not straightforward to properly set attributes/features
    # hence we use pyg_to_graph for now.
    # nxG = iG.to_networkx()
    # dsG = deepsnap.graph.Graph(nxG)
    return [dsG]
