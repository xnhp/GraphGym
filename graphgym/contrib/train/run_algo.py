from graphgym.register import register_train


# do not train any network but simply run some baseline algorithm


def run_alg(loggers, loaders, model, optimizer, scheduler):
    global comms
    from pytictoc import TicToc
    t = TicToc()
    # currently have only one batch containing all data
    mod = None
    pred = None
    for batch in loaders[0]:
        # compute modularity with baseline algorithm
        # consider bipartite / hyper later (use bipartite projection then, nx supports
        # that)
        nxG = batch.G[0]

        import networkx.algorithms.community as nx_comm
        t.tic()
        comms = nx_comm.greedy_modularity_communities(nxG)
        # NOTE this is crisp modularity!
        # todo for crisp assignments, should be the same?
        mod = nx_comm.modularity(nxG, comms)
        toctime = t.tocvalue()

    # ... thus also only one logger is relevant
    logger = loggers[0]

    logger.update_stats(true=None, pred=comms, loss=mod * -1, lr=0, time_used=toctime,
                        params=1, loss_main=mod * -1, loss_reg=0)
    logger.write_epoch(0, loader=loaders[0])

    for logger in loggers:
        logger.close()


register_train('run_alg', run_alg)
