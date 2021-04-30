from typing import Any

import torch
import torch.nn.functional as F
from contrib.loss.loss_utils import soft_modularity_on_device
from graphgym.config import cfg
from graphgym.register import register_loss


def soft_modularity_hyper_gg(pred, true, batch) -> [Any, Any]:
    # `true` would be supervised information (useless here)
    # `pred` would be node embeddings

    # following GG implementation, you have to do an additional check here
    if cfg.model.loss_fun != 'soft_modularity_hyper':
        raise Exception('loss fn should not have been called')

    raise NotImplementedError


register_loss('soft_modularity_hyper', soft_modularity_hyper_gg)


def soft_modularity_bip_gg(pred, _, batch) -> [Any, Any]:
    # TODO rename, this has nothing to do with bipartite
    # following GG implementation, you have to do an additional check here
    if cfg.model.loss_fun != 'soft_modularity_bipartite':
        raise Exception('loss fn should not have been called')

    # with a config like test-sbml-single.yaml there will only be a single batch
    # in a single split containing the entire graph.
    # potentially, training on batches (ego-networks?) could be thinkable for CD,
    # however we should do this as a sanity check and baseline.

    # we need: U (predictions), A (adjacency) {, node_degrees}
    # a meaningful class prediction encodes class membership probabilities.
    # instead of actually changing the output of the model, we do this here when
    # evaluating the loss
    pred_score = F.softmax(pred, dim=1)
    G = batch.G[0]

    mod = soft_modularity_on_device(G, pred_score)

    # want to minimise
    loss_mod = mod * -1  # todo create and call proper method in train/loss.py instead

    # collapse regularization as per [[tsitsulin_graph_2020]]
    # todo put this someplace else and make its usage configurable
    cluster_sizes = pred_score.sum(dim=0)
    num_nodes = torch.tensor(G.number_of_nodes()).to(torch.device(cfg.device))
    num_clusters = torch.tensor(cfg.dataset.num_communities).to(torch.float).to(
        torch.device(cfg.device))
    loss_reg = torch.linalg.norm(cluster_sizes) / num_nodes * torch.sqrt(num_clusters) - 1

    total_loss = loss_mod + cfg.model.collapse_lam * loss_reg

    return total_loss, pred_score, loss_mod, loss_reg


register_loss('soft_modularity_bipartite', soft_modularity_bip_gg)
