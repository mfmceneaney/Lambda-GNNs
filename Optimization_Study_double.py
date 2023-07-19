'''
USER SET VARIABLES
'''
import argparse
parser = argparse.ArgumentParser(description='PyTorch NF for Lambda events')

parser.add_argument('--Date', type=str, default="Undated",
                        help='Date to include in file names')
parser.add_argument('--num_epochs', type=int, default="2",
                        help='Number of epochs to train over')
parser.add_argument('--Distortion', type=float, default="0.2",
                        help='Range of distortion to add to MC')
args = parser.parse_args()
Date = args.Date
distort_value = args.Distortion
nepochs = args.num_epochs

import normflows as nf
from normflows import flows
## Standard libraries
import os
import math
import time
import numpy as np 

## Imports for plotting
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
import torch.optim as optim

import dgl #NOTE: for dgl.batch and dgl.unbatch
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_info, load_info, Subset

from tqdm import tqdm

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

#custom imports
from utils import load_graph_dataset, train, evaluate, GraphDataset, get_graph_dataset_info
from models import GIN, HeteroGIN
from NF_utils import Latent_data, create_latent_data, get_masked_affine, transform, transform_double, train,plot_loss, test,plot_9_histos, plot_UMAP_sidebyside,plot_UMAP_overlay
from GAN_utils import GAN_Input, GAN_Input_double

from numpy.random import default_rng
rng = default_rng()


#location for smearing matrix to be saved to
lambda_prefix = "/hpc/group/vossenlab/rck32/Lambda-GNNs/plots/hyperparameter_optimization"
string_distort_value_dot = str(distort_value)
string_distort_value = string_distort_value_dot.replace(".","_")

import optuna
from optuna.trial import TrialState


def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim",20,200)
    half_num_layers = trial.suggest_int("half_num_layers", 5,70)
    num_layers = 2 * half_num_layers
    lr = trial.suggest_float("learning rate", 1e-6,5e-3)
    file_prefix = lambda_prefix + "/loss_plots/num_layers_" + str(num_layers) + "_hiddendim_" + str(hidden_dim) + "_lr_" + str(lr)
    model_prefix = lambda_prefix + "/models/num_layers_" + str(num_layers) + "_hiddendim_" + str(hidden_dim) + "_lr_" + str(lr)

    '''                                              '''
    '''     SETTING UP LATENT SPACE REPRESENTATION   '''
    '''                                              '''

    # Data and MC both have the same prefix
    prefix = "/hpc/group/vossenlab/mfm45/.dgl/"

    # MC inside Lambda_train_matched_jobs_outbending_cache_bg50nA_7_28_22__pT_phi_theta_beta_chi2_pid_status__Normalized
    MCdataset = "Lambda_train_matched_jobs_outbending_cache_bg50nA_7_28_22__pT_phi_theta_beta_chi2_pid_status__Normalized"

    # Data inside data_jobs_rga_fall2018_7_28_22__pT_phi_theta_beta_chi2_pid_status__Normalized
    DATAdataset = "data_jobs_rga_fall2018_7_28_22__pT_phi_theta_beta_chi2_pid_status__Normalized"


    num_samples = 100
    MC_Graphs = GraphDataset(prefix+MCdataset)

    inputs = GAN_Input_double(MC_Graphs,distortion_range = (-distort_value,distort_value), distort = True, num_features = 20, num_sample_features = 12)

    # SETTING UP MC MODEL

    masked_affine_flows_train_MC = get_masked_affine(latent_dim = 12,hidden_dim = hidden_dim,num_layers = num_layers, alternate_mask = False)
    distribution_MC = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
    masked_affine_model_MC = nf.NormalizingFlow(q0=distribution_MC, flows=masked_affine_flows_train_MC)
    MC_model = masked_affine_model_MC.to(device)

    # TRAINING MC
    MC_loss, MC_full_loss = train(inputs, MC_model, distorted = False, num_epochs = nepochs,compact_num = 10, show_progress = False,lr = lr)
    try:
        plot_loss(MC_loss, label = "MC loss",save = True,save_loc = file_prefix + ".jpeg")
    except Exception as e:
        print(f"Caught exception: {e}\nContinuing...")
        raise optuna.exceptions.TrialPruned()
    MC_model.save(model_prefix + ".pth")
    # SETTING UP DATA MODEL

#     masked_affine_flows_train_distort = get_masked_affine(latent_dim = 12, hidden_dim = hidden_dim, num_layers = num_layers,alternate_mask = False)
#     distribution_distort = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
#     masked_affine_model_distort = nf.NormalizingFlow(q0=distribution_distort, flows=masked_affine_flows_train_distort)
#     distort_model = masked_affine_model_distort.to(device)

#     # TRAINING Distorted
#     distort_loss, distort_full_loss = train(inputs, distort_model, distorted = True, num_epochs = nepochs, compact_num = 10, show_progress = False,lr = lr)
#     plot_loss(distort_loss, label = "distort loss",save = True,save_loc = lambda_prefix + "/plots/NF_loss/distort_" + Date + "_double_features_" + string_distort_value + ".jpeg")

#     distort_model.save(lambda_prefix + "/models/NF_distort/distort_" + Date + "_double_features_" + string_distort_value + ".pth")

    # Testing MC
    return test(inputs, MC_model, data_type = "MC", return_loss = True)
#     # Testing DATA
#     test(inputs, distort_model, data_type = "MC distorted", distorted = True)

#     # normalized_MC = transform(inputs, MC_model)
#     normalized_distorted = transform_double(inputs, distort_model, distorted = True)

#     normalized_distorted_obj = Latent_data(normalized_distorted, torch.empty([]))
#     normalized_distorted_obj.set_batch_size(100)
#     full_pass_distorted = transform_double(normalized_distorted_obj, MC_model, reverse = False)

#     plot_fpd = torch.clone(full_pass_distorted)
#     plot_distorted_data = torch.clone(inputs.distorted_features)
#     plot_train_data = torch.clone(inputs.data)
#     for i in range(len(plot_fpd)):
#         for j in range(12):
#             if(np.isnan(plot_fpd[i,j])):
#                 plot_fpd[i,0] = 99999
#                 plot_train_data[i,0] = 99999
#                 plot_distorted_data[i,0] = 99999
#                 break
#     plot_fpd = plot_fpd[plot_fpd[:,0] != 99999]
#     plot_distorted_data = plot_distorted_data[plot_distorted_data[:,0] != 99999]
#     plot_train_data = plot_train_data[plot_train_data[:,0] != 99999]


#     fig, ((ax11,ax12,ax13),(ax21,ax22,ax23),(his1,his2,his3)) = plt.subplots(3,3,figsize = (12,12))
#     axlist = [ax11,ax12,ax13,ax21,ax22,ax23]
#     names = ["Proton pT", "Proton phi", "Proton theta", "Pion pT", "Pion phi", "Pion theta"]
#     fig.suptitle("Distorted vs fullpass with +/-" + string_distort_value_dot + " distortion")
#     for i in range(6):
#         x1 = torch.Tensor.numpy(plot_distorted_data[:,i])
#         y1 = torch.Tensor.numpy(plot_fpd[:,i])
#         axlist[i].hist2d(x1,y1,bins = 200, cmap = plt.cm.jet)
#         axlist[i].set_xlabel(names[i])
#     ax11.set_xlim(-1,1)
#     ax11.set_ylim(-1,1)
#     his1.hist(inputs.data[:,0], bins = 100,color = 'r')
#     his1.set_xlabel("MC")
#     his1.set_xlim(-1,1)
#     his2.hist(inputs.distorted_features[:,0], bins = 100,color = 'b')
#     his2.set_xlabel("distorted")
#     his2.set_xlim(-1,1)
#     his3.hist(full_pass_distorted[:,0], bins = 100,color = 'g')
#     his3.set_xlabel("fullpass")
#     his3.set_xlim(-1,1)

study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials = 100, timeout=14400)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy = False, states = [TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
#     fig.savefig(smearing_save_loc)