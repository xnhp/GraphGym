import networkx as nx
from graphgym.register import register_train

def groundtruth_fun(loggers, loaders, model, optimizer, scheduler):
    """
    A custom training module that does not train anything but reads the ground-truth
    labels supplied with the graph and computes their modularity.
    TODO: could probably have used the attribute `node_label` for this as expected by GG.
    :param loggers:
    :param loaders:
    :param model:
    :param optimizer:
    :param scheduler:
    :return:
    """
    # currently have only one batch containing all data
    mod_nx = None
    for batch in loaders[0]:
        nxG = batch.G[0]
        partition = batch.partition[0]
        mod_nx = nx.algorithms.community.modularity(nxG, partition)
        print("mod_nx", mod_nx)

    # ... thus also only one logger is relevant
    logger = loggers[0]

    logger.update_stats(true=None, pred=partition, loss=mod_nx*-1, lr=0, time_used=0,
                        params=1, loss_main=mod_nx*-1, loss_reg=0)
    logger.write_epoch(0, loader=loaders[0])

    for logger in loggers:
        logger.close()


register_train('groundtruth', groundtruth_fun)
