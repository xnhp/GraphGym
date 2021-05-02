import torch.nn as nn
from torch_geometric.nn import HypergraphConv
from graphgym.register import register_layer

# for GraphGym, need to wrap `MessagePassing` objects as such
# s.t. they take and produce `batch`es. This is pointless in
# the case of a single graph but we have to follow the API.

class HypergraphConvGG(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(HypergraphConvGG, self).__init__()
        self.model = HypergraphConv(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


register_layer('hyperconv', HypergraphConvGG)