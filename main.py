###############################
# Matthew McEneaney
# 6/24/21
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

def monitor(log_interval,filename="CPU_logs.txt"):
    with open(filename, 'w') as f:
        while shared_resource:
            f.write("CPU: %f RAM: %f" % (psutil.cpu_percent(log_interval), psutil.virtual_memory()[2]))
            f.write("\n")

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='PyTorch GIN for graph classification')
    parser.add_argument('--dataset', type=str, default="lambdas_big",
                        help='name of dataset (default: lambdas_big)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='Number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of model layers (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='Number of output MLP layers (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Number of hidden dimensions in model (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='Dropout rate for final layer (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling type over entire graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="max", choices=["sum", "average", "max"],
                        help='Pooling type over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--filename', type = str, default = "cpu_logs.txt",
                                        help='Output file for CPU/RAM monitoring')
    args = parser.parse_args()

    # Set up and seed devices
    torch.manual_seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Setup data and model
    train_dataloader, val_dataloader, num_classes, node_feature_dim = load_graph_dataset(batch_size=args.batch_size)
    print(args.num_layers, args.num_mlp_layers, node_feature_dim,
            args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
            args.neighbor_pooling_type, device)
    model = GIN(args.num_layers, args.num_mlp_layers, node_feature_dim,
            args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
            args.neighbor_pooling_type, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Train model
    utils.train(args, model, device, train_dataloader, val_dataloader, optimizer, scheduler, criterion, max_epochs)

    shared_resource = False

if __name__ == '__main__':
    # Define threads
    t1 = threading.Thread(target=monitor, name="monitor", args=(4,), daemon=True)
    t2 = threading.Thread(target=main, name="main")

    # Start threads
    t1.start()
    t2.start()
