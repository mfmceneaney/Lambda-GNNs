###############################
# Matthew McEneaney
# 7/8/21
###############################

from __future__ import absolute_import, division, print_function

# DGL Graph Learning Imports
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim

# Utility Imports
import argparse, math, datetime, os, psutil, threading

# Custom Imports
from utils import LambdasDataset, train, load_graph_dataset
from models import GIN

# Set key for monitor thread to check if main thread is done
shared_resource = True
lock = threading.Lock()

def monitor(log_interval,filename="cpu_logs.txt"):
    with open(filename, 'w') as f:
        while shared_resource:
            f.write("CPU: %f RAM: %f" % (psutil.cpu_percent(log_interval), psutil.virtual_memory()[2]))
            f.write("\n")

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='PyTorch GIN for graph classification')
    parser.add_argument('--dataset', type=str, default="lambdas_big",
                        help='name of dataset (default: lambdas_big) note: Needs to be in ~/.dgl')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: cuda:0)')
    parser.add_argument('--nworkers', type=int, default=0,
                        help='Number of dataloader workers (default: 0)')
    parser.add_argument('--batch', type=int, nargs=2, default=[256,257],
                        help='input batch size range for training (default: 256 257)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, nargs=2, default=[1e-1,1e-6],
                        help='Learning rate range (default: 1e-1 1e-5)')
    parser.add_argument('--step', type=int, nargs=2, default=[100,101],
                        help='Learning rate step size range (default: 100 101)')
    parser.add_argument('--gamma', type=float, nargs=2, default=[0.1,0.6],
                        help='Learning rate reduction factor range (default: 0.1 0.6)')
    parser.add_argument('--nlayers', type=int, nargs=2, default=[2,9],
                        help='Number of model layers range (default: 2 9)')
    parser.add_argument('--nmlp', type=int, nargs=2, default=[2,5],
                        help='Number of output MLP layers range (default: 2 5)')
    parser.add_argument('--hdim', type=int, nargs=2, default=[32,129],
                        help='Number of hidden dimensions in model range (default: 32 129)')
    parser.add_argument('--dropout', type=float, nargs=2, default=[0.0,0.5],
                        help='Dropout rate for final layer range (default: 0.0 0.5)')
    parser.add_argument('--gpooling', type=str, default="sum", choices=["sum", "mean"],
                        help='Pooling type over entire graph: sum or mean')
    parser.add_argument('--npooling', type=str, default="max", choices=["sum", "mean", "max"],
                        help='Pooling type over neighboring nodes: sum, mean or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--ntrials', type=int, default=100,
                        help='Number of study trials (default: 100)')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Max wait time for improvement in trials (default: 600)')
    parser.add_argument('--pruning', action="store_true",
                        help='Whether to use optuna pruner or not')
    parser.add_argument('--filename', type = str, default = "cpu_logs.txt",
                                        help='Output file for CPU/RAM monitoring')
    parser.add_argument('--verbose', action="store_true",
                                    help='Print messages and graphs')
    args = parser.parse_args()

    # Set up and seed devices
    torch.manual_seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Run optimization study
    utils.optimize(args)

    shared_resource = False

if __name__ == '__main__':
    # Define threads
    t1 = threading.Thread(target=monitor, name="monitor", args=(4,), daemon=True)
    t2 = threading.Thread(target=main, name="main")

    # Start threads
    t1.start()
    t2.start()
