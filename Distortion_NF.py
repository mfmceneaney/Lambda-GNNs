'''
USER SET VARIABLES
'''
import argparse
parser = argparse.ArgumentParser(description='PyTorch NF for Lambda events')

parser.add_argument('--Date', type=str, default="Undated",
                        help='Date to include in file names')
parser.add_argument('--Distortion', type=float, default=0.01,
                        help='Distortion range')
parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of epochs for training NF model')
args = parser.parse_args()
Date = args.Date
distort_value = args.Distortion
nepochs = args.num_epochs
hidden_dim = 79
num_layers = 94
lr = 0.00015


#location for smearing matrix to be saved to
lambda_prefix = "/hpc/group/vossenlab/rck32/Lambda-GNNs"
string_distort_value_dot = str(distort_value)
string_distort_value = string_distort_value_dot.replace(".","_")
smearing_save_loc = lambda_prefix + "/plots/NF_double_smear/" + Date + "/distort" + string_distort_value + ".jpeg"
smearing_save_loc_path = lambda_prefix + "/plots/NF_double_smear/" + Date
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

'''
FINISH THIS CODE HERE
'''
loss_path_date = lambda_prefix + "/plots/NF_loss/Double_" + Date
loss_path_MC = lambda_prefix + "/plots/NF_loss/Double_" + Date + "/MC"
loss_path_distort = lambda_prefix + "/plots/NF_loss/Double_" + Date + "/distort"
model_path_MC = lambda_prefix + "/models/NF_MC/MC_" + Date + "_double_features"
model_path_distort = lambda_prefix + "/models/NF_distort/distort_" + Date + "_double_features"

dir_list = [loss_path_date,smearing_save_loc_path,loss_path_MC,loss_path_distort,model_path_MC,model_path_distort]
for i in range(len(dir_list)):
    if(not os.path.isdir(dir_list[i])):
        os.mkdir(dir_list[i])

# SETTING UP MC MODEL

#Uncomment to train new model:
# masked_affine_flows_train_MC = get_masked_affine(latent_dim = 12,hidden_dim = hidden_dim,num_layers = num_layers, alternate_mask = False)
# distribution_MC = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
# masked_affine_model_MC = nf.NormalizingFlow(q0=distribution_MC, flows=masked_affine_flows_train_MC)
# MC_model = masked_affine_model_MC.to(device)

# # TRAINING MC
# MC_loss, MC_full_loss = train(inputs, MC_model, distorted = False, num_epochs = nepochs,compact_num = 10, show_progress = False,lr = lr)
# plot_loss(MC_loss, label = "MC loss",save = True,save_loc = lambda_prefix + "/plots/NF_loss/Double_" + Date + "/MC/distort_" + string_distort_value + ".jpeg")
# MC_model.save(lambda_prefix + "/models/NF_MC/MC_" + Date + "_double_features/MC_" + string_distort_value + ".pth")

#Using model we trained earlier:
masked_affine_flows_train_MC = get_masked_affine(latent_dim = 12,hidden_dim = hidden_dim,num_layers = num_layers, alternate_mask = False)
distribution_MC = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
masked_affine_model_MC = nf.NormalizingFlow(q0=distribution_MC, flows=masked_affine_flows_train_MC)
masked_affine_model_MC.load("models/NF_MC/MC_2023_07_19_double_features/MC_2023_07_19_double_features_0_1.pth")
MC_model = masked_affine_model_MC.to(device)

# # SETTING UP DATA MODEL
# masked_affine_flows_train_distort = get_masked_affine(latent_dim = 12, hidden_dim = hidden_dim, num_layers = num_layers,alternate_mask = False)
# distribution_distort = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
# masked_affine_model_distort = nf.NormalizingFlow(q0=distribution_distort, flows=masked_affine_flows_train_distort)
# distort_model = masked_affine_model_distort.to(device)

# # TRAINING Distorted
# distort_loss, distort_full_loss = train(inputs, distort_model, distorted = True, num_epochs = nepochs, compact_num = 10, show_progress = False,lr = lr)
# plot_loss(distort_loss, label = "distort loss",save = True,save_loc = lambda_prefix + "/plots/NF_loss/Double_" + Date + "/distort/distort" + string_distort_value + ".jpeg")
# distort_model.save(lambda_prefix + "/models/NF_distort/distort_" + Date + "_double_features/distort_" + string_distort_value + ".pth")

#Using model we trained earlier:
masked_affine_flows_train_distort = get_masked_affine(latent_dim = 12,hidden_dim = hidden_dim,num_layers = num_layers, alternate_mask = False)
distribution_distort = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
masked_affine_model_distort = nf.NormalizingFlow(q0=distribution_distort, flows=masked_affine_flows_train_distort)
masked_affine_model_distort.load(lambda_prefix + "/models/NF_distort/distort_" + Date + "_double_features/distort_" + string_distort_value + ".pth")
distort_model = masked_affine_model_distort.to(device)

# Testing MC
test(inputs, MC_model, data_type = "MC")
# Testing DATA
test(inputs, distort_model, data_type = "MC distorted", distorted = True)

# normalized_MC = transform(inputs, MC_model)
normalized_distorted = transform_double(inputs, distort_model, distorted = True)

normalized_distorted_obj = Latent_data(normalized_distorted, torch.empty([]))
normalized_distorted_obj.set_batch_size(100)
full_pass_distorted = transform_double(normalized_distorted_obj, MC_model, reverse = False)

plot_fpd = torch.clone(full_pass_distorted)
plot_distorted_data = torch.clone(inputs.distorted_features)
plot_train_data = torch.clone(inputs.data)
for i in range(len(plot_fpd)):
    for j in range(12):
        if(np.isnan(plot_fpd[i,j])):
            plot_fpd[i,0] = 99999
            plot_train_data[i,0] = 99999
            plot_distorted_data[i,0] = 99999
            break
plot_fpd = plot_fpd[plot_fpd[:,0] != 99999]
plot_distorted_data = plot_distorted_data[plot_distorted_data[:,0] != 99999]
plot_train_data = plot_train_data[plot_train_data[:,0] != 99999]


fig, ((ax11,ax12,ax13),(ax21,ax22,ax23),(ax31,ax32,ax33),(his1,his2,his3)) = plt.subplots(4,3,figsize = (12,15))
axlist = [ax21,ax22,ax23,ax31,ax32]
names = ["Proton pT", "Proton phi", "Proton theta", "Pion pT", "Pion phi", "Pion theta"]
fig.suptitle("Distorted vs fullpass with +/-" + string_distort_value_dot + " distortion_fix")
x1 = torch.Tensor.numpy(plot_distorted_data[:,0])
y1 = torch.Tensor.numpy(plot_fpd[:,0])
ax11.hist2d(x1,y1,bins = 200, cmap = plt.cm.jet)
ax11.set_xlabel("full pass vs distorted")

x2 = torch.Tensor.numpy(plot_distorted_data[:,0])
y2 = torch.Tensor.numpy(plot_train_data[:,0])
ax12.hist2d(x2,y2,bins = 200, cmap = plt.cm.jet)
ax12.set_xlabel("MC vs distorted")

x3 = torch.Tensor.numpy(plot_train_data[:,0])
y3 = torch.Tensor.numpy(plot_fpd[:,0])
ax13.hist2d(x3,y3,bins = 200, cmap = plt.cm.jet)
ax13.set_xlabel("full pass vs MC")
for i in range(5):
    x4 = torch.Tensor.numpy(plot_distorted_data[:,i+1])
    y4 = torch.Tensor.numpy(plot_fpd[:,i+1])
    axlist[i].hist2d(x4,y4,bins = 200, cmap = plt.cm.jet)
    axlist[i].set_xlabel(names[i+1])
# ax11.set_xlim(-1,1)
# ax11.set_ylim(-1,1)
his1.hist(inputs.data[:,0], bins = 100,color = 'r')
his1.set_xlabel("MC")
if((max(inputs.data[:,0]) > 10) or (min(inputs.data[:,0]) < -10)):
    his1.set_xlim(-1,1)
his2.hist(inputs.distorted_features[:,0], bins = 100,color = 'b')
his2.set_xlabel("distorted")
if((max(inputs.data[:,0]) > 10) or (min(inputs.data[:,0]) < -10)):
    his2.set_xlim(-1,1)
his3.hist(full_pass_distorted[:,0], bins = 100,color = 'g')
his3.set_xlabel("fullpass")
if((max(inputs.data[:,0]) > 10) or (min(inputs.data[:,0]) < -10)):
    his3.set_xlim(-1,1)
fig.savefig(smearing_save_loc)