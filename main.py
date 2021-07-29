###############################
# Matthew McEneaney
# COLABORATORY VERSION
# 7/29/21
###############################

from __future__ import absolute_import, division, print_function

# ML Imports
import matplotlib.pyplot as plt

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
from utils import LambdasDataset, load_graph_dataset, train, evaluate
from models import GIN, HeteroGIN

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='PyTorch GIN for graph classification')
    parser.add_argument('--dataset', type=str, default="gangelmc_10k_2021-07-22_noEtaOldChi2",
                        help='name of dataset (default: gangelmc_10k_2021-07-22_noEtaOldChi2)') #NOTE: Needs to be in ~/.dgl
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--nworkers', type=int, default=0,
                        help='Number of dataloader workers (default: 0)')
    parser.add_argument('--batch', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--step', type=int, default=10,
                        help='Learning rate step size (default: 20)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate reduction factor (default: 0.63)')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='Number of model layers (default: 3)')
    parser.add_argument('--nmlp', type=int, default=3,
                        help='Number of output MLP layers (default: 3)')
    parser.add_argument('--hdim', type=int, default=64,
                        help='Number of hidden dimensions in model (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.8,
                        help='Dropout rate for final layer (default: 0.8)')
    parser.add_argument('--gpooling', type=str, default="max", choices=["sum", "average"],
                        help='Pooling type over entire graph: sum or average')
    parser.add_argument('--npooling', type=str, default="max", choices=["sum", "average", "max"],
                        help='Pooling type over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--verbose', action="store_true",
                                    help='Print messages and graphs')
    # HeteroGIN Options
    parser.add_argument('--nfmlp', type=int, default=3,
                        help='Number of output MLP layers for HeteroGIN model (default: 3)')
    parser.add_argument('--hfdim', type=int, default=0,
                        help='Number of hidden final dimensions in HeteroGIN model (default: 0)')

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

    # Input dataset directory prefix option
    parser.add_argument('--prefix', type=str, default='/content/gdrive/My Drive/',
                        help='Prefix for where dataset is stored (default: /content/gdrive/My Drive/)')

    args = parser.parse_args()

    # Set up and seed devices
    torch.manual_seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Setup data and model
    train_dataloader, val_dataloader, nclasses, nfeatures = load_graph_dataset(dataset=args.dataset,prefix=args.prefix,
                                                    num_workers=args.nworkers, batch_size=args.batch)

    model = GIN(args.nlayers, args.nmlp, nfeatures,
            args.hdim, nclasses, args.dropout, args.learn_eps, args.npooling,
            args.gpooling).to(device)

    if args.hfdim > 0:
        nkinematics = 6 #TODO: Automate this assignment.
        model = HeteroGIN(args.nlayers, args.nmlp, nfeatures,
            args.hdim, nclasses, args.dropout, args.learn_eps, args.npooling,
            args.gpooling, nkinematics, args.hfdim, args.nfmlp).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    # Setup log directory
    try: os.mkdir(args.log)
    except FileExistsError: print('Directory:',args.log,'already exists!')

    # Train model
    train(args, model, device, train_dataloader, val_dataloader, optimizer, scheduler, criterion, args.epochs, dataset=args.dataset, prefix=args.prefix, log_dir=args.log, verbose=args.verbose)
    evaluate(model, device, dataset=args.dataset, prefix=args.prefix, log_dir=args.log, verbose=args.verbose)
    if args.verbose: plt.show()

if __name__ == '__main__':

    main()
