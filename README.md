# Lambdas-GNNs

Graph Neural Netowork implementations for Lambda Hyperon studies at CLAS12.

Check out [How Powerful Are Graph Neural Nets?](https://github.com/weihua916/powerful-gnns) for the GIN implementation and the paper on [arXiv:1810.00826](https://arxiv.org/abs/1810.00826).

Required python modules (all available with pip) are listed in `requirements.txt`.

# Installing the right version of PyTorch
Follow this link: [PyTorch Get Started Locallly](https://pytorch.org/get-started/locally/).

# Installing CUDA Python Libraries

Try installing a new python virtual environment with:
```
/apps/python/3.9.5/bin/python3.9 -m venv venv_cuda
source /full/path/to/venv_cuda/bin/activate
which python
deactivate
```

Also a good idea to put your venv packages first in your python path with:
```
export PYTHONPATH=/full/path/to/venv_cuda/lib/python*/site-packages/:$PYTHONPATH
```

Start an interactive GPU job (not entirely sure if this is necessary but it’s nice for checking your CUDA version):

```
srun -p gpu -c 2 --gres=gpu:1 --mem-per-cpu=8G --pty bash
```

Start your python3 cuda virtual environment:
```
source /full/path/to/venv_cuda/bin/activate
```

## Pytorch

Follow the instructions at [PyTorch Get Started Locallly](https://pytorch.org/get-started/locally/).
You should execute something like this:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## DGL

Try following instructions on the [DGL Get Started](https://www.dgl.ai/pages/start.html) page.

If that doesn’t work and you get an error like this:
```
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))': /wheels/repo.html
ERROR: Could not find a version that satisfies the requirement dgl-cu113 (from versions: none)
ERROR: No matching distribution found for dgl-cu113
```
try downloading locally whatever distribution you want from https://data.dgl.ai/wheels/repo.html.
Then transfer the downloaded distribution (e.g. with scp) to ifarm.

Then in your virtual environment and GPU job run (modify version if necessary:
```
pip install dgl-cu116 -f /path/to/distribution/you/just/uploaded
pip install dglgo -f https://data.dgl.ai/wheels/repo.html
```

#

Contact: matthew.mceneaney@duke.edu
