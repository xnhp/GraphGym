import networkx as nx
import torch
from config import cfg
from torch_sparse import SparseTensor

from train.loss import soft_modularity_simple


# library functions that are called from different clients should not
# be in the same module as the registration of a loss, since importing that
# in multiple places / scopes causes issues (GG bug)
def soft_modularity_on_device(graph: nx.Graph, pred_score):
    # if pipeline is not fully initialised yet, device might not have been
    # set to a concrete device (from originally "auto")
    device = "cpu" if cfg.device == "auto" else cfg.device
    # TODO avoidable performance cost converting to/from scipy
    adj_tens = SparseTensor.from_scipy(nx.to_scipy_sparse_matrix(graph)) \
        .to_dense() \
        .to(torch.float) \
        .to(torch.device(device))
    # todo call with degrees and num_edges precomputed
    # want to maximise
    return soft_modularity_simple(pred_score, adj_tens, transpose_U=False)
