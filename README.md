This is a fork of [GraphGym](https://github.com/snap-stanford/GraphGym) that enables support for unsupervised learning for the purpose of community detection. 

The changes made on top of the original framework can be viewed in the commit history of this repository.

The most notable changes are:

* Adjust the pipeline to unsupervised learning, in particular make adjustments s.t. ground-truth labels are not always expected
* Implement additional feature augmentations:
  - One-hot
  - Bethe-Hessian
  - Laplacian
* Load networks from different sources:
  - from `SBML` files
  - pre-generated `networkx` graphs given in `.npy` format
  - pre-generated `networkx` graphs given in `.gpickle` format
* Integrate the soft modularity as a loss function
* Add custom training modules (training loops)
  - additionally provide information of graph to loss function in order to compute modularity
  - do not train anything but invoke a baseline algorithm
  - do not train anything but read the ground-truth assignment labels of a given graph and compute the modularity of the parition they imply.
* Extend logging to print additional information on each epoch such as sizes of communities, modularity and value of regularisation term.
* Add additional configuration options
  - for community detection task
  - for generation of synthetic graphs
