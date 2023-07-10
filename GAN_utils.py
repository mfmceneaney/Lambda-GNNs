'''
imports
'''

import numpy as np
import torch
import torch.nn as nn

'''
GAN Classes and Functions
'''

'''
GAN_Input Class
'''

class GAN_Input():
    def __init__(self, GraphDataset, num_features = 14,num_sample_features = 6, batch_size = 100, distortion_range = (-0.1,0.1), distort = True, shuffle = True, sidebands = False, sideband_cut = 1.14):
        self.GraphDataset = GraphDataset
        self.num_features = num_features
        self.num_sample_features = num_sample_features
        self.batch_size = batch_size
        self.num_events = len(self.GraphDataset.graphs)
        self.max_iter = int(self.num_events / self.batch_size)
        # self.data order:
        # proton pT, phi, theta, pion pT, phi, theta, label[0->7]
        self.data = torch.zeros(self.num_events,self.num_features)
        self.data += -9999
        self.create_data_tensor()
        if(sidebands):
            self.sideband_cut()
        if(shuffle):
            self.shuffle()
        if(distort):
            self.distort(distortion_range = distortion_range)
    def set_batch_size(self,batch_size):
        self.batch_size = batch_size
        self.max_iter = int(self.num_events / self.batch_size)
    def create_data_tensor(self):
        for i in range(self.num_events):
            event = self.GraphDataset.graphs[i].ndata['data']
            label = self.GraphDataset.labels[i]
            found_proton = False
            found_pion = False
            if not ((0.8 in event[:,5]) and (-0.6 in event[:,5])):
                continue
            #loop over each particle in graph
            for j in range(event.size()[0]):
                particle = event[j]
                #catch protons
                if particle[5] == 0.8:
                    for k in range(3):
                        self.data[i][k] = particle[k]
                    found_proton = True

                #catch pi-
                if particle[5] == -0.6:
                    for k in range(3):
                        self.data[i][k + 3] = particle[k]
                    found_pion = True

                #stop looking after lambda products found
                if(found_proton and found_pion):
                    #store label with data so when we shuffle they stick together
                    for k in range(8):
                        self.data[i][k + 6] = label[k]
                    break
                    
        self.data = self.data[self.data[:,0] > -9998]
        self.num_events = len(self.data)
        self.max_iter = int(self.num_events / self.batch_size)
        self.MC_min = self.data.min()
        self.MC_max = self.data.max()
    def sideband_cut(self):
        self.data = self.data[self.data[:,7] > 1.14]
    def shuffle(self):
        self.data = self.data[torch.randperm(self.data.size()[0])]
    def change_input(self, in_data):
        self.data = in_data
        self.num_events = len(in_data)
        self.max_iter = int(self.num_events / self.batch_size)
    def distort(self, index = 0, distortion_range = (-1,1)):
        #Grab random numbers
        distortions = torch.rand(self.data.shape[0],self.data.shape[1])
        #Trying to make distortions between -0.2 and 0.2 now
        distortions = (distortions * (distortion_range[0] - distortion_range[1])) + distortion_range[1]
        #Keep only numbers in the proton pT index
        distortions[:,1:] = 0
        self.distorted_features = torch.clone(self.data) + distortions
        self.distort_min = self.distorted_features.min()
        self.distort_max = self.distorted_features.max()
    def normalize(self, distorted = False):
        if(distorted):
            data = torch.clone(self.distorted_features)
        else:
            data = torch.clone(self.data)
        input_max = data.max()
        input_min = data.min()
        if(distorted):
            self.distorted_features = (data - input_min) / (input_max - input_min)
        else:
            self.data = (data - input_min) / (input_max - input_min)
#     def unnormalize(self, distored = False):
    def sample(self,iteration = 0, distorted = False):
        #0 index iterations - the "first" iteration is with iteration = 0
        # Calculate the first index we want to take from training data (rest of data is directly after)
        begin = iteration * self.batch_size
        # initialize
        samples = torch.zeros(self.batch_size, self.num_sample_features)
        #loop over consecutive tensors, save to return tensor
        if distorted:
            for i in range(self.batch_size):
                samples[i] = self.distorted_features[begin + i][:self.num_sample_features]
        else:
            for i in range(self.batch_size):
                samples[i] = self.data[begin + i][:self.num_sample_features]
        return samples
    
'''
GAN Generator
'''

#from PyTorch-GAN github
class Generator(nn.Module):
    def __init__(self, latent_dim = 6):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 160, normalize=False),
            *block(160, 80),
            *block(80, 40),
            *block(40, latent_dim)
        )

    def forward(self, z):
        out = self.model(z)
        return out

'''
GAN Discriminator
'''

class Discriminator(nn.Module):
    def __init__(self, latent_dim = 6):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 160),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(160, 80),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(80,40),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(40,latent_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        validity = self.model(input)
        return validity