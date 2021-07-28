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
from utils import LambdasDataset, train, load_graph_dataset, optimization_study
from models import GIN

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='PyTorch GIN for graph classification')
    parser.add_argument('--dataset', type=str, default="gangelmc_10k_2021-07-22_noEtaOldChi2",
                        help='name of dataset (default: gangelmc_10k_2021-07-22_noEtaOldChi2) note: Needs to be in ~/.dgl')
    parser.add_argument('--device', type=str, default='cpu',
                        help='which device to use if any (default: \'cpu\')')
    parser.add_argument('--nworkers', type=int, default=0,
                        help='Number of dataloader workers (default: 0)')
    parser.add_argument('--batch', type=int, nargs=2, default=[256,257],
                        help='input batch size range for training (default: 256 257)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, nargs=2, default=[1e-4,1e-2],
                        help='Learning rate range (default: 1e-5 1e-2)')
    parser.add_argument('--step', type=int, nargs=2, default=[10,101],
                        help='Learning rate step size range (default: 100 101)')
    parser.add_argument('--gamma', type=float, nargs=2, default=[0.1,1.0],
                        help='Learning rate reduction factor range (default: 0.1 1.0)')
    parser.add_argument('--nlayers', type=int, nargs=2, default=[4,5],
                        help='Number of model layers range (default: 4 5)')
    parser.add_argument('--nmlp', type=int, nargs=2, default=[3,4],
                        help='Number of output MLP layers range (default: 3 4)')
    parser.add_argument('--hdim', type=int, nargs=2, default=[128,129],
                        help='Number of hidden dimensions in model range (default: 128 129)')
    parser.add_argument('--dropout', type=float, nargs=2, default=[0.5,1.0],
                        help='Dropout rate for final layer range (default: 0.5 1.0)')
    parser.add_argument('--gpooling', type=str, default="sum", choices=["sum", "mean"],
                        help='Pooling type over entire graph: sum or mean')
    parser.add_argument('--npooling', type=str, default="max", choices=["sum", "mean", "max"],
                        help='Pooling type over neighboring nodes: sum, mean or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--ntrials', type=int, default=100,
                        help='Number of study trials (default: 100)')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Max wait time for improvement in trials (default: 1800)')
    parser.add_argument('--pruning', action="store_true",
                        help='Whether to use optuna pruner or not')
    parser.add_argument('--verbose', action="store_true",
                                    help='Print messages and graphs')
    # Output directory option
    parser.add_argument('--log', type=str, default='logs/unique/',
                        help='Log directory for histograms (default: logs/<options used>/)')
    
    # Early stopping options
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='Minimum change threshold for early stopping (default: 0.0)')
    parser.add_argument('--cumulative_delta', action='store_true',
                        help='Use cumulative change since last patience reset as opposed to last event (default: false)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs to wait for early stopping (default: 10)')

    args = parser.parse_args()

    # Set up and seed devices
    torch.manual_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Setup log directory
    try: os.mkdir(args.log)
    except FileExistsError: print('Directory:',args.log,'already exists!')

    # Run optimization study
    optimization_study(args)

if __name__ == '__main__':

    main()
