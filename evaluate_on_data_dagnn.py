###############################
# Matthew McEneaney
# 7/28/21
###############################

from __future__ import absolute_import, division, print_function

# ML Imports
import matplotlib.pyplot as plt

# DGL Graph Learning Imports
from dgl.dataloading import GraphDataLoader

# PyTorch Imports
import torch

# Utility Imports
import argparse, os

# Custom Imports
from utils import get_graph_dataset_info, load_graph_dataset, evaluate, evaluate_on_data
from models import GIN, HeteroGIN, Classifier, Discriminator
import models

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
    parser.add_argument('--nlayers', type=int, default=3,
                        help='Number of model layers (default: 3)')
    parser.add_argument('--nmlp', type=int, default=3,
                        help='Number of output MLP layers (default: 3)')
    parser.add_argument('--hdim', type=int, default=64,
                        help='Number of hidden dimensions in model (default: 64)')
    parser.add_argument('--nmlp_head', type=int, default=3,
                        help='Number of output MLP layers in classifier/discriminator (default: 3)')
    parser.add_argument('--hdim_head', type=int, default=64,
                        help='Number of hidden dimensions in classifier/discriminator (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.8,
                        help='Dropout rate for final layer (default: 0.8)')
    parser.add_argument('--gpooling', type=str, default="max", choices=["sum", "mean", "max"],
                        help='Pooling type over entire graph: sum or mean')
    parser.add_argument('--npooling', type=str, default="max", choices=["sum", "mean", "max"],
                        help='Pooling type over neighboring nodes: sum, mean or max')
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
    parser.add_argument('--log', type=str, default='eval/',
                        help='Log directory for histograms (default: eval/)')

    # Model load directory
    parser.add_argument('--path', type=str, default='logs',
                        help='Directory from which to load model (default: logs)')
    parser.add_argument('--name', type=str, default='model', #NOTE: Corresponds to `--save_path` argument in training.
                        help='Name for file in which to save model (default: model)')

    # Input dataset directory prefix option
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix for where dataset is stored (default: ~/.dgl/)')

    # Input dataset train/val split
    parser.add_argument('--split', type=float, default=0.1,
                        help='Fraction of dataset to use for evaluation (default: 0.1)')

    # Input dataset train/val max total events
    parser.add_argument('--max_events', type=float, default=1e5,
                        help='Max number of train/val events to use (default: 1e5)')

    args = parser.parse_args()

    # Set up and seed devices
    torch.manual_seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Setup data and model
    nclasses, nfeatures, nfeatures_edge = get_graph_dataset_info(dataset=args.dataset, prefix=args.prefix)

    _model = GIN(args.nlayers, args.nmlp, nfeatures,
            args.hdim, args.hdim, args.dropout, args.learn_eps, args.npooling,
            args.gpooling).to(device)
    _classifier = MLP(args.nmlp_head, args.hdim, args.hdim_head, nclasses).to(device)
    print("DEBUGGING: LOADING: ",os.path.join(args.path,args.name+'_model_weights'))#DEBUGGING
    print("DEBUGGING: LOADING: ",os.path.join(args.path,args.name+'_classifier_weights'))#DEBUGGING
    _model.load_state_dict(torch.load(os.path.join(args.path,args.name+'_model_weights'),map_location=device))
    _classifier.load_state_dict(torch.load(os.path.join(args.path,args.name+'_classifier_weights'),map_location=device))

    model = models.Concatenate([ _model, _classifier])

    # if args.hfdim > 0:
    #     nkinematics = 6 #TODO: Automate this assignment.
    #     model = HeteroGIN(args.nlayers, args.nmlp, nfeatures,
    #         args.hdim, nclasses, args.dropout, args.learn_eps, args.npooling,
    #         args.gpooling, nkinematics, args.hfdim, args.nfmlp).to(device)
    
    model.eval()

    # Setup log directory
    try: os.mkdir(args.log)
    except FileExistsError: print('Directory:',args.log,'already exists!')

    # Train model
    evaluate_on_data(model, device, dataset=args.dataset, prefix=args.prefix, split=args.split, log_dir=args.log, verbose=args.verbose, batch_size=args.batch, num_workers=args.nworkers)
    if args.verbose: plt.show()

if __name__ == '__main__':

    main()
