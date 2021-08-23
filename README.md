# Lambdas-GNNs

Graph Neural Network implementations for Lambda Hyperon studies at CLAS12.

Check out [How Powerful Are Graph Neural Nets?](https://github.com/weihua916/powerful-gnns) for the Graph Isomorphism Network (GIN) implementation and the paper on [arXiv:1810.00826](https://arxiv.org/abs/1810.00826).

Required python modules (all available with pip) are listed in `requirements.txt`.

## Getting Started

To read data from ROOT we use [Uproot](https://uproot.readthedocs.io/en/latest/) which interfaces with numpy/awkward arrays.  The documentation there is fairly comprehensive with nice examples.  You can also read through `create_dataset.py` to get an idea of how this can be implemented.

We use the [Deep Graph Library (DGL)](https://www.dgl.ai) which has lots of good resources and examples with papers on different classification tasks.  It also runs with Tensorflow, PyTorch and Apache MXNet.  Here we implement in PyTorch.  (Make sure you install the GPU-version if you want to use a GPU to train your network.)

To train your network you can either write your own training loops or use a wrapper package such as [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) which is what we implement here.  The documentation is pretty straightforward, but you can read through the `train()` and `evaluate()` functions in `utils.py` to get a sense of how this might be implemented in practice.

To run a distributed parameter optimization search we use [Optuna](https://optuna.readthedocs.io/en/stable/) which also has pretty decent documentation (or you can read through the `optimize()` function in `utils.py`).  Results are stored in an SQL database if the search is distributed so make sure you brush up on some SQL syntax.

#

Contact: matthew.mceneaney@duke.edu
