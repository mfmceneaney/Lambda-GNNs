'''
USER SET VARIABLES
'''
import argparse
parser = argparse.ArgumentParser(description='PyTorch NF for Lambda events')

parser.add_argument('--Date', type=str, default="Undated",
                        help='Date to include in file names')
parser.add_argument('--Distortion', type=float, default=0.01,
                        help='Distortion range')
parser.add_argument('--Sigma', type=float, default=0.1,
                        help='Standard deviation to distort from')
parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of epochs for training NF model')
parser.add_argument('--shift', type=bool, default=False,
                        help='Tells input object if it should shift the data instead of smearing')
parser.add_argument('--shift_val', type=float, default=0.2,
                        help='Amount to shift the pT distribution')
parser.add_argument('--extra_info', type=str, default="",
                        help='Extra info to add to file names')
parser.add_argument('--multimodal', type=bool, default=False,
                        help='Tells input object if it should use multimodal gaussians')
parser.add_argument('--alternate_mask', type=bool, default=False,help='Tells input object if it should alternate bit mask')
args = parser.parse_args()
Date = args.Date
distort_value = args.Distortion
sigma = args.Sigma
nepochs = args.num_epochs
shift = args.shift
shift_val = args.shift_val
multimodal = args.multimodal
alternate = args.alternate_mask
hidden_dim = 79
num_layers = 94
lr = 0.00015
old_date = "2023_07_19"

#location for smearing matrix to be saved to
extra_info = args.extra_info
lambda_prefix = "/hpc/group/vossenlab/rck32/Lambda-GNNs"

string_distort_value_dot = str(shift_val)
string_distort_value = string_distort_value_dot.replace(".","_")

string_sigma_dot = str(sigma)
string_sigma = string_sigma_dot.replace(".","_")

#If we are distorting from random distribution, use the standard deviation as distortion value, not the shift
if(not shift):
    string_distort_value = string_sigma
    string_distort_dot = string_sigma_dot
smearing_save_loc = lambda_prefix + "/plots/NF_double_smear/" + Date + "/distort" + string_distort_value + extra_info + ".jpeg"
smearing_save_loc_path = lambda_prefix + "/plots/NF_double_smear/" + Date
histos_save_loc = lambda_prefix + "/plots/NF_double_histos/" + Date + "/distort" + string_distort_value + extra_info + ".jpeg"
histos_save_loc_path = lambda_prefix + "/plots/NF_double_histos/" + Date

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
from GAN_utils import GAN_Input

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

inputs = GAN_Input(MC_Graphs,distortion_range = (-distort_value,distort_value), distort = True, num_features = 20, num_sample_features = 12, shift = shift, shift_val = shift_val,sigma = sigma, double = True)

loss_path_date = lambda_prefix + "/plots/NF_loss/Double_" + Date
loss_path_MC = lambda_prefix + "/plots/NF_loss/Double_" + Date + "/MC"
loss_path_distort = lambda_prefix + "/plots/NF_loss/Double_" + Date + "/distort"
model_path_MC = lambda_prefix + "/models/NF_MC/MC_" + Date + "_double_features"
model_path_distort = lambda_prefix + "/models/NF_distort/distort_" + Date + "_double_features"

dir_list = [loss_path_date,smearing_save_loc_path,loss_path_MC,loss_path_distort,model_path_MC,model_path_distort,histos_save_loc_path]
for i in range(len(dir_list)):
    if(not os.path.isdir(dir_list[i])):
        os.mkdir(dir_list[i])
'''
MULTIMODAL MODEL TRAINING
'''
# SETTING UP MC MODEL
if(multimodal):
    # SETTING UP MC MODEL
    # SETTING UP DATA MODEL
    masked_affine_flows_train_distort = get_masked_affine(latent_dim = 12, hidden_dim = hidden_dim, num_layers = num_layers,alternate_mask = (not double))
    # distribution_distort = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = True)
    distribution_distort = nf.distributions.GaussianMixture(4,12,loc = [[0.2,-0.75,-0.2,-0.2,-0.15,-0.2,0.2,-0.75,-0.2,-0.2,-0.15,-0.2],[0.2,0.75,-0.2,-0.2,-0.15,-0.2,0.2,0.75,-0.2,-0.2,-0.15,-0.2],[0.2,-0.75,-0.2,-0.2,0.15,-0.2,0.2,-0.75,-0.2,-0.2,0.15,-0.2],[0.2,0.75,-0.2,-0.2,0.15,-0.2,0.2,0.75,-0.2,-0.2,0.15,-0.2]], trainable = True)
    masked_affine_model_distort = nf.NormalizingFlow(q0=distribution_distort, flows=masked_affine_flows_train_distort)
    distort_model = masked_affine_model_distort.to(device)

    # TRAINING Distorted
    distort_loss, distort_full_loss = train(inputs, distort_model, distorted = True, num_epochs = nepochs, compact_num = 10, show_progress = False,lr = lr)
    plot_loss(distort_loss, label = "distort loss",save = True,save_loc = lambda_prefix + "/plots/NF_loss/Double_" + Date + "/distort/distort" + string_distort_value + extra_info + ".jpeg")
    distort_model.save(lambda_prefix + "/models/NF_distort/distort_" + Date + "_double_features/distort_" + string_distort_value + extra_info + ".pth")

    # Uncomment to train new model:
    masked_affine_flows_train_MC = get_masked_affine(latent_dim = 12,hidden_dim = hidden_dim,num_layers = num_layers, alternate_mask = (not double))
    # distribution_MC = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = True)
    masked_affine_model_MC = nf.NormalizingFlow(q0=distribution_distort, flows=masked_affine_flows_train_MC)
    MC_model = masked_affine_model_MC.to(device)

    # TRAINING MC
    MC_loss, MC_full_loss = train(inputs, MC_model, distorted = False, num_epochs = nepochs,compact_num = 10, show_progress = False,lr = lr)
    plot_loss(MC_loss, label = "MC loss",save = True,save_loc = lambda_prefix + "/plots/NF_loss/Double_" + Date + "/MC/distort_" + string_distort_value + extra_info + ".jpeg")
    MC_model.save(lambda_prefix + "/models/NF_MC/MC_" + Date + "_double_features/MC_" + string_distort_value +  extra_info +".pth")
    # #Using model we trained earlier:
    # masked_affine_flows_train_MC = get_masked_affine(latent_dim = 12,hidden_dim = hidden_dim,num_layers = num_layers, alternate_mask = False)
    # distribution_MC = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
    # masked_affine_model_MC = nf.NormalizingFlow(q0=distribution_MC, flows=masked_affine_flows_train_MC)
    # masked_affine_model_MC.load("models/NF_MC/MC_2023_07_19_double_features/MC_2023_07_19_double_features_0_1.pth")
    # MC_model = masked_affine_model_MC.to(device)

    '''
    TESTING RN: CHANGED MC GAUSSIAN TO TRAINABLE AND MADE DISTORT DIST TO MC
    '''



    # #Using model we trained earlier:
    # masked_affine_flows_train_distort = get_masked_affine(latent_dim = 12,hidden_dim = hidden_dim,num_layers = num_layers, alternate_mask = False)
    # distribution_distort = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
    # masked_affine_model_distort = nf.NormalizingFlow(q0=distribution_distort, flows=masked_affine_flows_train_distort)
    # masked_affine_model_distort.load(lambda_prefix + "/models/NF_distort/distort_" + old_date + "_double_features/distort_" + string_distort_value + ".pth")
    # distort_model = masked_affine_model_distort.to(device)
    
    
'''
SINGLE MODE TRAINING
'''
if(not multimodal):
    # Uncomment to train new model:
    masked_affine_flows_train_MC = get_masked_affine(latent_dim = 12,hidden_dim = hidden_dim,num_layers = num_layers, alternate_mask = alternate)
    distribution_MC = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
    masked_affine_model_MC = nf.NormalizingFlow(q0=distribution_MC, flows=masked_affine_flows_train_MC)
    MC_model = masked_affine_model_MC.to(device)

    # TRAINING MC
    MC_loss, MC_full_loss = train(inputs, MC_model, distorted = False, num_epochs = nepochs,compact_num = 10, show_progress = False,lr = lr)
    plot_loss(MC_loss, label = "MC loss",save = True,save_loc = lambda_prefix + "/plots/NF_loss/Double_" + Date + "/MC/distort_" + string_distort_value + extra_info + ".jpeg")
    MC_model.save(lambda_prefix + "/models/NF_MC/MC_" + Date + "_double_features/MC_" + string_distort_value +  extra_info +".pth")

    # #Using model we trained earlier:
    # masked_affine_flows_train_MC = get_masked_affine(latent_dim = 12,hidden_dim = hidden_dim,num_layers = num_layers, alternate_mask = False)
    # distribution_MC = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
    # masked_affine_model_MC = nf.NormalizingFlow(q0=distribution_MC, flows=masked_affine_flows_train_MC)
    # masked_affine_model_MC.load("models/NF_MC/MC_2023_07_19_double_features/MC_2023_07_19_double_features_0_1.pth")
    # MC_model = masked_affine_model_MC.to(device)

    # SETTING UP DATA MODEL
    masked_affine_flows_train_distort = get_masked_affine(latent_dim = 12, hidden_dim = hidden_dim, num_layers = num_layers,alternate_mask = alternate)
    distribution_distort = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
    masked_affine_model_distort = nf.NormalizingFlow(q0=distribution_distort, flows=masked_affine_flows_train_distort)
    distort_model = masked_affine_model_distort.to(device)

    # TRAINING Distorted
    distort_loss, distort_full_loss = train(inputs, distort_model, distorted = True, num_epochs = nepochs, compact_num = 10, show_progress = False,lr = lr)
    plot_loss(distort_loss, label = "distort loss",save = True,save_loc = lambda_prefix + "/plots/NF_loss/Double_" + Date + "/distort/distort" + string_distort_value + extra_info + ".jpeg")
    distort_model.save(lambda_prefix + "/models/NF_distort/distort_" + Date + "_double_features/distort_" + string_distort_value + extra_info + ".pth")

    # #Using model we trained earlier:
    # masked_affine_flows_train_distort = get_masked_affine(latent_dim = 12,hidden_dim = hidden_dim,num_layers = num_layers, alternate_mask = False)
    # distribution_distort = nf.distributions.DiagGaussian(inputs.num_sample_features, trainable = False)
    # masked_affine_model_distort = nf.NormalizingFlow(q0=distribution_distort, flows=masked_affine_flows_train_distort)
    # masked_affine_model_distort.load(lambda_prefix + "/models/NF_distort/distort_" + old_date + "_double_features/distort_" + string_distort_value + ".pth")
    # distort_model = masked_affine_model_distort.to(device)

# Testing MC
test(inputs, MC_model, data_type = "MC", show_progress = False)
# Testing DATA
test(inputs, distort_model, data_type = "MC distorted", distorted = True, show_progress = False)

# normalized_MC = transform(inputs, MC_model)
normalized_distorted = transform_double(inputs, distort_model, distorted = True, show_progress = False)

normalized_distorted_obj = Latent_data(normalized_distorted, torch.empty([]))
normalized_distorted_obj.set_batch_size(100)
full_pass_distorted = transform_double(normalized_distorted_obj, MC_model, reverse = False, show_progress = False)

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
fig.suptitle("Distorted vs fullpass with +/-" + string_distort_value_dot + " distortion" +extra_info)
x1 = torch.Tensor.numpy(plot_distorted_data[:,0])
y1 = torch.Tensor.numpy(plot_fpd[:,0])
ax11.hist2d(x1,y1,bins = 200, cmap = plt.cm.jet)
ax11.set_xlabel("full pass vs distorted")

x2 = torch.Tensor.numpy(plot_distorted_data[:,0])
y2 = torch.Tensor.numpy(plot_train_data[:,0])
ax12.hist2d(x2,y2,bins = 200, cmap = plt.cm.jet)
ax12.set_xlabel("MC vs distorted")

x3 = torch.Tensor.numpy(plot_fpd[:,0])
y3 = torch.Tensor.numpy(plot_train_data[:,0])
ax13.hist2d(x3,y3,bins = 200, cmap = plt.cm.jet)
ax13.set_xlabel("MC vs fullpass")
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

# fig3, a = plt.subplots(1,1)
# fig3.suptitle("Smearing Matrix at +/-" + string_distort_value_dot + " distortion")
# a.hist2d(x1,y1,bins = 200, cmap = plt.cm.jet)
# a.set_xlabel("full pass vs distorted")
# fig3.savefig(lambda_prefix + "/one_plot.jpeg")

fig2, ((h1, h2, h3),(h4, h5, h6),(h7,h8,h9),(h10,h11,h12),(h13,h14,h15),(h16,h17,h18)) = plt.subplots(6,3,figsize = (12,23))
hlist = [h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18]
fig.suptitle("Kinematic Histograms with +/-" + string_distort_value_dot + " distortion"+ extra_info)
for i in range(6):
    hlist[i].hist(inputs.data[:,i],bins = 100, color = 'r')
    hlist[i].set_xlabel("MC " + names[i])
    hlist[i + 6].hist(full_pass_distorted[:,i],bins = 100, color = 'g')
    hlist[i + 6].set_xlabel("fullpass " + names[i])
    hlist[i + 12].hist(inputs.distorted_features[:,i],bins = 100,color = 'b')
    hlist[i + 12].set_xlabel("distorted " + names[i])
    if((max(inputs.data[:,i]) > 10) or (min(inputs.data[:,i]) < -10)):
        hlist[i].set_xlim(-1,1)
    if((max(inputs.distorted_features[:,i]) > 10) or (min(inputs.distorted_features[:,i]) < -10)):
        hlist[i+6].set_xlim(-1,1)
    if((max(full_pass_distorted[:,i]) > 10) or (min(full_pass_distorted[:,i]) < -10)):
        hlist[i+12].set_xlim(-1,1)
fig2.savefig(histos_save_loc)
    