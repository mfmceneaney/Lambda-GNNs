'''
IMPORTS
'''

import normflows as nf
from normflows import flows
## Standard libraries
import os
import math
import time
import numpy as np 

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
import torch.optim as optim

torch.manual_seed(42)

import dgl #NOTE: for dgl.batch and dgl.unbatch
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_info, load_info, Subset

import umap
reducer = umap.UMAP();
from tqdm import tqdm

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

#custom imports
from utils import load_graph_dataset, train, evaluate, GraphDataset, get_graph_dataset_info
from models import GIN, HeteroGIN

from numpy.random import default_rng
rng = default_rng()

'''
Latent_data Class
    
    Info:
        Purpose:
            -Simplify use of tensors filled with training/testing data for use in NF models
        Benefits:
            -Collects all needed attributes of data into one object
            -Implements necessary functions:
                -sampling
    Reference:
        Constructor:
            -Parameters:
                *in_tensor: tensor filled with data, organized in form: `tensor[num_events][dimensionality]`
            -Methods:
                *__init__: initializes attributes that are available at construction time
                *set_batch_size: sets the batch size and calculates number of batches
                    +parameters: 
                        ~batch_size: (int) number of samples per training iteration
                *sample: calls proper sampling function depending on random parameter
                    +parameters: 
                        ~iteration: (int) current iteration of training/testing; 
                        ~random: (bool) conduct random or fixed order sampling
                *sample_fixed: takes the next `batch_size` events from the dataset and returns them in a tensor
                    +parameters:
                        ~iteration: (int) see sample reference
                    +returns:
                        ~samples: tensor of samples
                *sample_random: uses np random generation to return a random event from the dataset
                    +returns:
                        ~samples: tensor of samples
'''
class Latent_data:
    def __init__(self, in_tensor,labels):
        self.data = in_tensor
        self.num_events = in_tensor.size()[0]
        self.latent_size = in_tensor.size()[1]
        self.labels = labels
    def set_batch_size(self,batch_size):
        self.batch_size = batch_size
        self.max_iter = int(self.num_events / self.batch_size)
    def set_mass(self, mass):
        self.mass = mass
    def sample(self,iteration = 0, random = False, _give_labels = False):
        if(random):
            return self.sample_random(give_labels = _give_labels)
        else:
            return self.sample_fixed(iteration,give_labels = _give_labels)
    def sample_fixed(self,iteration,give_labels = False):
        #0 index iterations - the "first" iteration is with iteration = 0
        # Calculate the first index we want to take from training data (rest of data is directly after)
        begin = iteration * self.batch_size
        # initialize
        samples = torch.zeros(self.batch_size, self.latent_size)
        labels = torch.zeros(self.batch_size, 1)
#         print(f"labels max (inside sample_fixed): {labels.max()}")
        #loop over consecutive tensors, save to return tensor
        if(give_labels):
            for i in range(self.batch_size):
                samples[i] = self.data[begin + i]
                labels[i] = self.labels[begin+i]
            return samples,labels
        else:
            for i in range(self.batch_size):
                samples[i] = self.data[begin + i]
            return samples
    def sample_random(self,labels = False):
        indices = rng.integers(low=0, high=self.num_events, size=self.batch_size)
        samples = torch.zeros(self.batch_size,self.latent_size)
        for index in range(len(indices)):
            samples[index] = self.data[indices[index]]
        return samples

'''
create_latent_data Function

    Info:
        Purpose:
            -Abstraction for creating testing/training data object from Graphdataset
    Reference:
        Parameters:
            -dataset_directory: (string) name of directory where dataset is located
            -extractor (GIN [see Lambda-GNNs/models.py])
            -optional:
                *prefix: (string) path to directory where dataset is stored
                *split: (float) fraction of events to use for training (1 - split is fraction for testing)
                *max_events: (int) total number of events to use for both training and testing
                *num_samples: (int) number of events to include in each training/testing batch
                *mode: (string) options are training or testing - tells function to use either split or (1 - split)
        Returns:
            (Latent_data) object with dataset loaded, preconfigured with batch size
'''
def create_latent_data(dataset_directory, extractor, prefix = "/hpc/group/vossenlab/mfm45/.dgl/", split = 0.8, max_events = 140000, num_samples = 250, mode = "default",shuffle = True):
    val_split = (1 - split) / 2
    if(mode == "test"):
        data_range = range(int(split*max_events),int((val_split + split)*max_events))
    elif(mode == "train"):
        data_range = range(0, int(split*max_events))
    elif(mode == "val"):
        data_range = range(int((val_split + split)*max_events),max_events)
    elif(mode == "default"):
        print(f"No mode given, defaulting to training\n")
        data_range = range(0, int(split*max_events))
    else:
        raise Exception("Invalid mode: {mode}\nPlease use either \"train,\" or \"test\" ", mode)
    dataset = GraphDataset(prefix+dataset_directory)
    dataset.load()
    if(shuffle):
        dataset.shuffle()
    dataset = Subset(dataset,data_range)
    dgl_batch = dgl.batch(dataset.dataset.graphs[dataset.indices.start:dataset.indices.stop])
    labels = dataset.dataset.labels[dataset.indices.start:dataset.indices.stop,0].clone().detach().float().view(-1, 1)
    mass = dataset.dataset.labels[dataset.indices.start:dataset.indices.stop,1].clone().detach().float()
    dgl_batch = dgl_batch.to(device)
    labels = labels.to(device)
    latent = extractor.get_latent_repr(dgl_batch).detach().cpu()
    latent_obj = Latent_data(latent,labels)
    latent_obj.set_batch_size(num_samples)
    latent_obj.set_mass(mass)
    return latent_obj
    
'''
get_masked_affine function
    Info
        Purpose:
            -Creates new list of affine coupling layers for use in NormalizingFlow() programs 
        Sources:
            -normalizing-flows python package: https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/
            -realNVP model: https://openreview.net/pdf?id=HkpbnH9lx
            -masked affine coupling example: https://github.com/VincentStimper/normalizing-flows/blob/master/examples/image.ipynb
    Reference
        Variables:
            -b: (torch.tensor) bitmask filled with alternating 0s and 1s to split dataspace into 2
            -masked_affine_flows: (list of nf.flows()) list of coupling layers
            -s: (MLP) scale function
            -t: (MLP) translation function
        Returns:
            -list of coupling layers
'''
def get_masked_affine(num_layers = 32):
    #mask
    b = torch.ones(71)
    for i in range(b.size()[0]):
        if i % 2 == 0:
            b[i] = 0
    masked_affine_flows = []
    for i in range(num_layers):
        s = nf.nets.MLP([71, 142, 142, 71])
        t = nf.nets.MLP([71, 142, 142, 71])
        if i % 2 == 0:
            masked_affine_flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            masked_affine_flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    return masked_affine_flows


'''
transform function
    Info
        -Performs invertible functions on an input to flow from on distribution to another
        -Forward flow transforms a gaussian to a latent target
        -Inverse flow transforms a latent target to a gaussian
    Reference
        -Parameters:
            *in_data: (Latent_data object) data you want to transform
            *model: (nf.NormalizingFlow) trained NF flow model to use for transformation 
            *reverse: (bool) direction of flow (defaults to reverse - latent to gaussian)
    
        -Returns:
            *data_tensor: (torch.tensor) transformed tensor with same size as in_data.data
'''
def transform(in_data, model, reverse = True):
    data_tensor = torch.zeros_like(in_data.data)
    model.eval()
    with torch.no_grad():
        for it in tqdm(range(in_data.max_iter), position = 0, leave=True):
            test_samples = in_data.sample(iteration = it)
            test_samples = test_samples.to(device)
            if(reverse):
                output_batch = model.inverse(test_samples)
            else:
                output_batch = model.forward(test_samples)
            for i in range(in_data.batch_size):
                data_tensor[it*in_data.batch_size + i] = output_batch[i]
    return data_tensor

'''
train function
    Info:
        -Runs a training loop to train NF model
    Reference:
        -Parameters:
            *in_data: (Latent_data object) data you want to train model on
            *model: (nf.NormalizingFlow) model to train
        -Returns:
            *loss_hist: (list of floats) list of loss values for each batch for plotting
'''

def train(in_data, model, val = False,val_data = Latent_data(torch.empty(10000,71), torch.empty(10000,71)), num_epochs = 1):
    # train the MC model
    if(val):
        val_data.set_batch_size(int(floor(val_data.num_events / in_data.max_iter)))
        val_loss_hist = np.array([])
    model.train()
    loss_hist = np.array([])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    for i in range(num_epochs):
        with tqdm(total=in_data.max_iter, position=0, leave=True) as pbar:
            for it in tqdm(range(in_data.max_iter), position = 0, leave=True):
                optimizer.zero_grad()
                #randomly sample the latent space
                samples = in_data.sample(iteration = it)
                samples = samples.to(device)
                loss = model.forward_kld(samples)
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                # Log loss
                if~(torch.isnan(loss)):
                    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
            if(val):
                val_samples = val_data.sample(iteration = it)
                val_samples = val_samples.to(device)
                val_loss = model.forward_kld(val_samples)
                if~(torch.isnan(val_loss)):
                    val_loss_hist = np.append(val_loss_hist, val_loss.to('cpu').data.numpy())
    if(val):
        return loss_hist, val_loss_hist
    else:
        return loss_hist

'''
plot_loss function
    Info:
        -Plots loss as a function of batch number
    Reference:
        -Parameters:
            *loss_hist: (list of floats) losses indexed by batch
            *save: (bool) if true, saves plot as image
            *save_loc: (string) path to save plot
            *label: (string) data label for legend
'''

def plot_loss(loss_hist,plot_val = False, val_loss_hist = np.array([]),save = False, save_loc = "plots/loss.jpeg", label = "loss"):
    fig = plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label=label)
    if(plot_val):
        plt.plot(val_loss_hist, label = "val")
    plt.legend()
    plt.show()
    print(f"Lowest Loss: {loss_hist.min()}")
    if save:
        fig.savefig(save_loc)
        
'''
test function
    Info:
        -Runs a testing loop to check performance of model
    Reference:
        -Parameters:
            *in_data: (Latent_data object) data you want to test model on
            *model: (nf.NormalizingFlow) model to test
            *data_type: (string) title for data to include in print statement stating loss
        -Effect:
            *Prints average loss
'''        

def test(in_data, model, data_type = "none"):
    model.eval()
    test_loss = 0
    counted_batches = 0
    with torch.no_grad():
        for it in tqdm(range(in_data.max_iter), position = 0, leave=True):
            test_samples = in_data.sample(iteration = it)
            test_samples = test_samples.to(device)
            new_loss = model.forward_kld(test_samples)
            if(not math.isnan(new_loss)):
                test_loss += new_loss
                counted_batches += 1
        if(data_type == "none"):
            print(f"average loss: {test_loss/counted_batches}")
        else:
            print(f"{data_type} average loss: {test_loss/counted_batches}")

'''
plot_9_histos function
    Info:
        -Plots 9 different 1D histograms of different dimensions of data using matplotlib subplots
    Reference:
        -Parameters:
            *data_tensor: (torch.tensor) data to plot
            *color: (string) color to make the histograms
            *bins: (int) number of bins to use
            *description: (string) super title of subplots
        -Effect:
            *Plots 9 different histograms in 3x3 grid
'''            

def plot_9_histos(data_tensor, color,bins = 150, description = "none"):
    histos, ((h11,h12,h13),(h21,h22,h23),(h31,h32,h33)) = plt.subplots(3,3, figsize = (10,10))
    if description == "none":
        histos.suptitle("Several 1D Histos")
    else:
        histos.suptitle(f"Several 1D Histos {description}")
    hlist = [h11,h12,h13,h21,h22,h23,h31,h32,h33]
    for i in range(len(hlist)):
        hlist[i].hist(data_tensor[:,i], bins=150,color=color);
    plt.show()

'''
plot_UMAP_sidebyside function
    Info:
        -Plots UMAP projection of two different datasets in side-by-side subplots
    Reference:
        -Parameters:
            *left_data: (torch.tensor or list) dataset for left plot in x,y coordinates
            *right_data: (torch.tensor or list) dataset for right plot in x,y coordinates
            *left_color: (string) color of left plot points
            *right_color: (string) color of right plot points
            *marker: (string) shape of marker for matplotlib
            *description: (string) heading for plot, format is `{description} projection with UMAP`
            *figsize: (tuple) dimensions for plot
            *left_description: (string) heading for left plot
            *right_description: (string) heading for right plot
            *marker_size: (int) size of points on plot
        -Effect:
            Plots side-by-side projections w/matplotlib
'''    
    
def plot_UMAP_sidebyside(left_data, right_data, left_color, right_color, marker = 'o', description = "none", figsize = (16,5), left_description = "none", right_description = "none",marker_size = 2, save="False", save_loc = "plots/UMAP_untitled.jpeg"):
    fig, (ax11, ax12) = plt.subplots(1,2, figsize = figsize)
    fig.suptitle(f"{description} projection with UMAP")
    ax11.plot(left_data[:,0],left_data[:,1],marker,markersize=marker_size,color = left_color)
    if not (left_description == "none"):
        ax11.set_title(left_description)
    ax12.plot(right_data[:,0],right_data[:,1],marker,markersize=marker_size,color=right_color)
    if not (right_description == "none"):
        ax12.set_title(right_description)
    plt.show()
    if save:
        fig.savefig(save_loc)
    
'''
plot_UMAP_overlay function
    Info:
        -Plots UMAP projection of two different datasets on one plot, one on top of the other
        -Uses lower opacity to show different datasets through one another (no blocking)
        -Best used alongside `plot_UMAP_sidebyside` to show where both latent spaces are separately, then on top of each other
    Reference:
        -Parameters:
            *left_data: (torch.tensor or list) dataset for left plot in x,y coordinates
            *right_data: (torch.tensor or list) dataset for right plot in x,y coordinates
            *left_color: (string) color of left plot points
            *right_color: (string) color of right plot points
            *marker: (string) shape of marker for matplotlib
            *description: (string) heading for plot, format is `{description} projection with UMAP`
            *figsize: (tuple) dimensions for plot
            *left_description: (string) heading for left plot
            *right_description: (string) heading for right plot
            *marker_size: (int) size of points on plot
            *alpha: (float; 0 < alpha < 1) opacity of datapoints; recommendation: 0.1
        -Effect:
            Plots side-by-side projections w/matplotlib
'''  
    
def plot_UMAP_overlay(left_data, right_data, left_color, right_color, marker = 'o', description = "none", left_description = "none", right_description = "none",marker_size = 2,alpha = 0.1, save="False", save_loc = "plots/UMAP_untitled.jpeg"):
    fig, ax = plt.subplots()
    ax = ax.twinx()
    if(description == "none"):
        fig.suptitle(f"UMAP Projection Overlay")
    else:
        fig.suptitle(f"UMAP Projection of {description}")
    #only put label if one is passed as a parameter
    if not (left_description == "none"):
        ax.plot(left_data[:,0],left_data[:,1],marker,markersize=marker_size, alpha=alpha,c=left_color, label = left_description)
    else:
        ax.plot(left_data[:,0],left_data[:,1],marker,markersize=marker_size, alpha=alpha,c=left_color)
    #only put label if one is passed as a parameter
    if not (right_description == "none"):
        ax.plot(right_data[:,0], right_data[:,1], marker, markersize=marker_size, alpha = alpha, c=right_color, label = right_description)
    else:
        ax.plot(right_data[:,0], right_data[:,1], marker, markersize=marker_size, alpha = alpha, c=right_color)
    if not ((left_description == "none") and (right_description == "none")):
        leg = fig.legend(title = "Key")
        for lh in leg.legend_handles:
            lh.set_alpha(1)
    plt.show()
    if save:
        fig.savefig(save_loc)