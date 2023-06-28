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
            *val: (bool) if true, also performs validation and returns validation histogram
            *val_data: (Latent_data) validation data
            *num_epochs: (int) number of times to run over the whole training dataset
            *compact_num: (int) number of iterations to average over before producing compact histogram
        -Returns:
            *compact_hist: (list of floats) list of loss values below a certain threshold, averaged, for readable plotting
            *full_loss_hist: (list of floats) list of all loss values for each batch
            *compact_hist_val: (list of floats) same as compact_hist but for validation
            *full_val_loss_hist: (list of floats) same as full_loss_hist but for validation
'''

def train(in_data, model, val = False,val_data = Latent_data(torch.empty(10000,71), torch.empty(10000,71)), num_epochs = 1, compact_num = 20):
    # train the MC model
    if(val):
        val_data.set_batch_size(int(np.floor(val_data.num_events / in_data.max_iter)))
        val_loss_hist = np.array([])
        full_val_loss_hist = np.array([])
    model.train()
    loss_hist = np.array([])
    full_loss_hist = np.array([])
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
                    full_loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
                    if(loss < 1000):
                        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
                if(val):
                    val_samples = val_data.sample(iteration = it)
                    val_samples = val_samples.to(device)
                    val_loss = model.forward_kld(val_samples)
                    if~(torch.isnan(val_loss)):
                        full_val_loss_hist = np.append(val_loss_hist, val_loss.to('cpu').data.numpy())
                        if(val_loss < 1000):
                            val_loss_hist = np.append(val_loss_hist, val_loss.to('cpu').data.numpy())
                            
    #
    # This section of code exists solely to create more readable histograms
    #
    running_ttl = 0
    compact_hist = np.array([])
    j = 0
    for i in range(loss_hist.size):
        if(j != (i // compact_num)):
            compact_hist = np.append(compact_hist,running_ttl / compact_num)
            running_ttl = 0
        j = i // compact_num
        running_ttl += loss_hist[i]
    if(val):
        running_ttl_val = 0
        compact_hist_val = np.array([])
        j = 0
        for i in range(val_loss_hist.size):
            if(j != (i // compact_num)):
                compact_hist_val = np.append(compact_hist_val,running_ttl_val / compact_num)
                running_ttl_val = 0
            j = i // compact_num
            running_ttl_val += val_loss_hist[i]
        return compact_hist, compact_hist_val, full_loss_hist, full_val_loss_hist
    else:
        return compact_hist, loss_hist

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
        
        
'''
NFClassifier class
    Info:
        -Class for creating a simple classifier that can be used for classifying latent representations of Lambda events
    Reference:
        -Parameters:
            *input_size: (int, default = 71) size of feature space of data
            *num_classes: (int, default = 2) number of classes for the classifier to pick between
            *hidden_dim: (int, default = 256) dimension of hidden layers
            *num_layers: (int, default = 3) number of layers to use in classifier
        -Functions:
            *forward: performs forward pass of classifier and returns classification in tuple
'''

class NFClassifier(nn.Module):
    """
    Classifier for normalized tensors
    """
    def __init__(self, input_size=71, num_classes=2, hidden_dim = 256, num_layers = 5):
        super(NFClassifier, self).__init__()
        self.layer = nn.Sequential()
        for i in range(num_layers):
            if(i == 0):
                self.layer.append(
                nn.Linear(input_size, hidden_dim)
                )
                self.layer.append(
                    nn.ReLU(inplace=True)
                )
            elif(i == num_layers - 1):
                self.layer.append(
                nn.Linear(hidden_dim, num_classes)
                )
            else:
                self.layer.append(
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.layer.append(
                    nn.ReLU(inplace=True)
                )
        self.name = "Classifier"
        
    def forward(self, h):
        c = self.layer(h)
        return c
    
    # @property
    def name(self):
        """
        Name of model.
        """
        return self.name
    
    
'''
train_classifier function
'''

def train_classifier(train_data, classifier, criterion, optimizer, val = True, val_data = Latent_data(torch.empty(10000,71), torch.empty(10000,71)), num_epochs = 1):
    loss_hist = np.array([])
    if(val):
        val_data.set_batch_size(int(np.floor(val_data.num_events / train_data.max_iter)))
        val_loss_hist = np.array([])
    for i in range(num_epochs):
        epoch_hist = np.array([])
        val_epoch_hist = np.array([])
        with tqdm(total=train_data.max_iter, position=0, leave=True) as pbar:
            for it in tqdm(range(train_data.max_iter), position = 0, leave=True):
                optimizer.zero_grad()
                #randomly sample the latent space
                samples, labels = train_data.sample(iteration = it, _give_labels = True)
                samples = samples.to(device)
                labels = (labels.type(torch.LongTensor)).to(device)
                # forward + backward + optimize
                outputs = classifier(samples)
                loss = criterion(outputs, labels[:,0])
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                # Log loss
                if~(torch.isnan(loss)):
                    epoch_hist = np.append(epoch_hist, loss.to('cpu').data.numpy())
                if(val):
                    #validation
                    val_samples, val_labels = val_data.sample(iteration = it, _give_labels = True)
                    val_samples = val_samples.to(device)
                    val_labels = (val_labels.type(torch.LongTensor)).to(device)
                    val_outputs = classifier(val_samples)
                    val_loss = criterion(val_outputs, val_labels[:,0])
                    val_epoch_hist = np.append(val_epoch_hist, val_loss.to('cpu').data.numpy())
        loss_hist = np.append(loss_hist, epoch_hist.mean())
        if(val):
            val_loss_hist = np.append(val_loss_hist, val_epoch_hist.mean())

    print('Finished Training')
    if(val):
        return loss_hist, val_loss_hist
    else:
        return loss_hist
    
'''
test_classifier_MC function

'''
def test_classifier_MC(test_data, classifier):
    outputs = torch.empty(test_data.num_events,2)    
    with tqdm(total=test_data.max_iter, position=0, leave=True) as pbar:
        for it in tqdm(range(test_data.max_iter), position = 0, leave=True):
            #randomly sample the latent space
            samples, labels = test_data.sample(iteration = it, _give_labels = True)
            samples = samples.to(device)
            # forward + backward + optimize
            output_batch = classifier(samples)
            for i in range(test_data.batch_size):
                outputs[it*test_data.batch_size + i] = output_batch[i]
    test_Y     = test_data.labels.clone().detach().float().view(-1, 1).to("cpu")
    probs_Y = torch.softmax(outputs, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1,1)
    test_acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
    print(f"Accuracy: {test_acc * 100}")    
    
    
'''
test_classifier_data function
    Info
        -Function to classify Lambda events from data where labels are not given
    Reference
        -Parameters
            *test_data: (Latent_data) data to classify
            *classifier: (NFClassifier) trained classification model to classify data
        -Returns
            *argmax_Y: (tensor) predicted labels according to classifier
'''
    
def test_classifier_data(test_data, classifier):
    outputs_data = torch.empty(test_data.num_events,2)
    #Converting normalized DATA to classifier output
    with tqdm(total=test_data.max_iter, position=0, leave=True) as pbar:
        for it in tqdm(range(test_data.max_iter), position = 0, leave=True):
            #randomly sample the latent space
            samples, labels = test_data.sample(iteration = it, _give_labels = True)
            samples = samples.to(device)
            # forward + backward + optimize
            output_batch = classifier(samples)
            for i in range(test_data.batch_size):
                outputs_data[it*test_data.batch_size + i] = output_batch[i]
    probs_data = torch.softmax(outputs_data, 1)
    argmax_Y = torch.max(probs_data, 1)[1].view(-1,1)
    return argmax_Y

'''
plot_classified function
    Info
        -plots three mass spectra - signal, background and the full mass spectrum
        -useful to plot the classification made by a classifier
        -requires classes and masses and they must be in same order
    Reference
        -Parameters:
            *masses: (array-like) list of all masses of Lambdas in a dataset
            *classes: (array-like) list of 1s and 0s - 1 means yes lambda, 0 means no lambda
            *label: (string) combined histogram label
            *save: (bool, default = False) tells function if it should save an image of the plot to directory
            *save_loc: (string) tells function where to save plot image to
            *figsize: (tuple of ints) size of whole subplot canvas
            *bins: (int, default = 100) number of bins for each histogram
        -Effect:
            *plots signal, background, and combined histograms, and saves the plot if desired
'''

def plot_classified(masses, classes, label = "none",save = False, save_loc = "plots/default.jpeg", figsize = (18,4), bins = 100):
    num_total = int(classes.size()[0])
    num_signal = int(classes.sum())
    signal = np.zeros(num_signal)
    bg = np.zeros(num_total-num_signal)
    bg_count, signal_count = 0, 0
    for i in range(num_total):
        if classes[i]:
            signal[signal_count] = masses[i]
            signal_count+= 1
        else:
            bg[bg_count] = masses[i]
            bg_count += 1
    histos, (h1,h2,h3) = plt.subplots(1,3, figsize = figsize)
    h1.hist(signal, bins = bins, label = "signal", color = "b")
    h2.hist(bg, bins = bins, label = "background", color = "xkcd:orange")
    if(label != "none"):
        h3.hist(training_data_MC.mass, bins = bins, label = label)
    else:
        h3.hist(training_data_MC.mass, bins = bins)

    leg = histos.legend(title = "Key")
    histos.show()
    if(save):
        histos.savefig(save_loc)