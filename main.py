###############################
# Matthew McEneaney
# 7/8/21
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
from models import GIN

# # Set key for monitor thread to check if main thread is done
# shared_resource = True
# lock = threading.Lock()

# def monitor(log_interval,filename="cpu_logs.txt"):
#     with open(filename, 'w') as f:
#         while shared_resource:
#             f.write("CPU: %f RAM: %f" % (psutil.cpu_percent(log_interval), psutil.virtual_memory()[2]))
#             f.write("\n")

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
                        help='Number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--step', type=int, default=10,
                        help='Learning rate step size (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate reduction factor (default: 0.1)')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='Number of model layers (default: 2)')
    parser.add_argument('--nmlp', type=int, default=3,
                        help='Number of output MLP layers (default: 3)')
    parser.add_argument('--hdim', type=int, default=64,
                        help='Number of hidden dimensions in model (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for final layer (default: 0.2)')
    parser.add_argument('--gpooling', type=str, default="max", choices=["sum", "average"],
                        help='Pooling type over entire graph: sum or average')
    parser.add_argument('--npooling', type=str, default="max", choices=["sum", "average", "max"],
                        help='Pooling type over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
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

    # Setup data and model
    train_dataloader, val_dataloader, nclasses, nfeatures = load_graph_dataset(dataset=args.dataset,
                                                    num_workers=args.nworkers, batch_size=args.batch)

    model = GIN(args.nlayers, args.nmlp, nfeatures,
            args.hdim, nclasses, args.dropout, args.learn_eps, args.npooling,
            args.gpooling).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    # Train model
    train(args, model, device, train_dataloader, val_dataloader, optimizer, scheduler, criterion, args.epochs, dataset=args.dataset, verbose=args.verbose)
    evaluate(model, device, dataset=args.dataset, verbose=args.verbose)
    if args.verbose: plt.show()

    shared_resource = False

if __name__ == '__main__':
    # # Define threads
    # t1 = threading.Thread(target=monitor, name="monitor", args=(4,), daemon=True)
    # t2 = threading.Thread(target=main, name="main")

    # # Start threads
    # t1.start()
    # t2.start()

    main()
