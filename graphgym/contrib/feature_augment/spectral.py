import networkx
import networkx as nx
import numpy as np
import scipy
import torch
from graphgym.config import cfg
from graphgym.register import register_feature_augment


def laplacian_eigenvectors(graph, **kwargs):
    nxG = graph.G
    L = nx.laplacian_matrix(nxG)
    import numpy
    eigvals, eigvecs = numpy.linalg.eig(L.todense())
    # eigvecs is of shape [n, n]
    eigvecs_sorted = [x for _, x in sorted(zip(eigvals, eigvecs), key=lambda p: p[0])]
    # dropping the leading (eigenvalue 0)
    eigvecs_sorted = eigvecs_sorted[1:]
    # take the leading k eigenvectors
    feature_dim = kwargs['feature_dim']  # given by config file?
    eigvecs_take = eigvecs_sorted[:feature_dim]
    eigvecs_take = np.array(eigvecs_take).squeeze()
    # l-th row defines the features of node l, we output features for all nodes,
    # hence just need to transpose
    return torch.tensor(eigvecs_take.transpose())


register_feature_augment('node_laplacian', laplacian_eigenvectors)


def bethe_hessian_eigenvectors(graph, **kwargs):
    nxG: networkx.graph.Graph
    nxG = graph.G

    adj = nx.adjacency_matrix(nxG)

    degrees = np.array([deg for _, deg in list(nxG.degree)])
    degrees = np.asarray(adj.sum(axis=1), dtype=np.float64).flatten()
    r = np.sqrt((degrees ** 2).mean() / degrees.mean() - 1)
    n = adj.shape[0]
    eye = scipy.sparse.eye(n, dtype=np.float64)
    D = scipy.sparse.spdiags(degrees, [0], n, n, format='csr')
    bethe_hessian = (r ** 2 - 1) * eye - r * adj + D
    _, node_vecs = scipy.sparse.linalg.eigsh(bethe_hessian, k=cfg.dataset.num_communities,
                                             which="SA")

    tens = torch.tensor(node_vecs)
    return tens


register_feature_augment('node_bethe_hessian', bethe_hessian_eigenvectors)
