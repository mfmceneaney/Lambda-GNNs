"""
 Script for reading hipo2root ROOT files into masked numpy arrays
 and storing in ROOT and/or as a DGL dataset.
 Matthew McEneaney
 26 Jul. 2021
"""

from __future__ import absolute_import, division, print_function

# ROOT imports
import uproot as ur
import uproot3 as ur3

# ML Imports
import awkward as ak
import numpy as np
from numpy import ma
import torch
import dgl
from dgl.data import DGLDataset
import matplotlib.pyplot as plt

# Utility imports
import math, datetime, os, time
from utils import LambdasDataset

# Set constants:
pids = {
    "g":22,
    "e":11,
    "diquark":45, # Still not sure what this is...
    "pi0":111,
    "pip":211,
    "pim":-211,
    "k0":311,
    "kp":321,
    "km":-321,
    "p":2212,
    "n":2112,
    "L0":3122
}

masses = {
    "g":0.0,
    "e":5.11e-4,
    "diquark":0.0, # Still not sure what this is...
    "pi0":0.13495,
    "pip":0.13957,
    "pim":0.13957,
    "k0":0.4937,
    "kp":0.4976,
    "km":0.4976,
    "p":0.9389,
    "n":0.9396,
    "L0":1.1156
}

pid_replacements = {
    45:0.0,
    22:0.0,#0.1,
    111:0.2,
    211:0.5,#0.25,
    -211:-1,#0.3
    311:0.35,
    321:0.4,
    -321:0.45,
    2112:0.5,
    -2212:0.6,
    2212:1,#0.7,
    -11:0.9,#0.8,
    11:-0.5,
}

# Declare functions
def replace_pids(arr,pid_i):
    '''Arguments: arr   - masked ndarray with dtype=float
                  pid_i - last depth index for pids in arr
       Replace pids in given array roughly following scheme described in arxiv:1810.05165.
    '''
    
    if 'int' in str(arr.dtype):
        print(" *** ERROR *** array passed to replace_pids should not have dtype==int")
        return
    
    mask = ~arr[:,:,pid_i].mask
    for key in pid_replacements:
        arr[mask,pid_i] = np.where(arr[mask,pid_i]==key,
                                  pid_replacements[key],
                                  arr[mask,pid_i])

# Open TTrees with Uproot # Check out: https://uproot.readthedocs.io/en/latest/basic.html
# Note: 'clas12' is default key name when using hipo2root command line tool

# Loop variables
use_pids = True
step_size = 100
write_interval = 100 # How many iterations to wait before writing array to root
min_index = 0
max_index = step_size * write_interval
file_num = 0
write_counter = 0
max_events = 10**4+1 # +16000# * 10**4 + 1
max_events_string = str(int(max_events//10**3))+"k_" if max_events < 10**6 else str(int(max_events//10**6))+"m_"
data   = []
target = []
mass   = []
counter = 0
targetD = 100 # Max # of particles / event

# Lambda fraction variables
lambda_counter  = 0
other_counter   = 0
lambda_fraction = 0.5
lambda_label    = 1

# Require electron proton and pion reconstructed in event
strict = True

# I/O Variables
dataset_name = "gangelmc_"+max_events_string+datetime.datetime.now().strftime("%F") #NOTE: For naming output directory for root files and for naming DGL dataset
files  = [{"/home/mfmce/clas12work/train/small/*.root":"clas12"}]
outdir = "/work/clas12/users/mfmce/ROOT/"
num_workers = 2**3 # 2**3 is best on 4 or 8 cores, 2**5 is really slow.

start_date = time.strftime('%X')
start_time = time.time()

for batch in ur.iterate(files,step_size=step_size,num_workers=num_workers):
    
    # Get target array
    mc_pid      = ak.to_numpy(ak.pad_none(batch['MC_Lund_pid'],targetD,axis=1))
    targetp     = ma.array([lambda_label if pids["L0"] in arr else 0 for arr in mc_pid],mask=[False for arr in mc_pid])

    # Get data arrays
#     run     = ak.to_numpy(batch['RUN_Config_event'])
    masses  = ak.to_numpy(ak.pad_none(batch['REC_Kinematics_mass'],targetD,axis=1))
    Q2      = ak.to_numpy(ak.pad_none(batch['REC_Kinematics_Q2'],targetD,axis=1))
    W       = ak.to_numpy(ak.pad_none(batch['REC_Kinematics_W'],targetD,axis=1))
    x       = ak.to_numpy(ak.pad_none(batch['REC_Kinematics_x'],targetD,axis=1))
    y       = ak.to_numpy(ak.pad_none(batch['REC_Kinematics_y'],targetD,axis=1))
    z       = ak.to_numpy(ak.pad_none(batch['REC_Kinematics_z'],targetD,axis=1))
    xF      = ak.to_numpy(ak.pad_none(batch['REC_Kinematics_xF'],targetD,axis=1))
    px      = ak.to_numpy(ak.pad_none(batch['REC_Particle_px'],targetD,axis=1))
    py      = ak.to_numpy(ak.pad_none(batch['REC_Particle_py'],targetD,axis=1))
    pz      = ak.to_numpy(ak.pad_none(batch['REC_Particle_pz'],targetD,axis=1))
    pid     = ak.to_numpy(ak.pad_none(batch['REC_Particle_pid'],targetD,axis=1))
    chi2    = ak.to_numpy(ak.pad_none(batch['REC_Particle_chi2pid'],targetD,axis=1))
    status  = ak.to_numpy(ak.pad_none(batch['REC_Particle_status'],targetD,axis=1))
    charge  = ak.to_numpy(ak.pad_none(batch['REC_Particle_charge'],targetD,axis=1))
    beta    = ak.to_numpy(ak.pad_none(batch['REC_Particle_beta'],targetD,axis=1))
        
    # Compute some more useful arrays # Could also add theta and vertices -> phi and vT vz
    mask      = ~px[:,:].mask
    pT        = ma.array(np.zeros(np.shape(px)),mask=~mask)
    pT[mask]  = np.add(px[mask]**2,py[mask]**2)
    p2        = ma.array(np.zeros(np.shape(px)),mask=~mask)
    p2[mask]  = np.add(pT[mask]**2,pz[mask]**2)
    theta       = ma.array(np.zeros(np.shape(px)),mask=~mask)
    theta[mask] = np.divide(pT[mask],pz[mask])
    phi       = ma.array(np.zeros(np.shape(px)),mask=~mask)
    phi[mask] = np.arctan(py[mask],px[mask])
    eta       = ma.array(np.zeros(np.shape(px)),mask=~mask)
    eta[mask] = np.arctanh(np.divide(pz[mask],np.sqrt(p2[mask])))
    
    # Get data array and move eventwise axis to be first
    datap   = np.moveaxis(ma.array([pT,phi,eta,chi2,beta,status,charge,pid,theta,Q2,W,x,y,z,xF]),[0,1,2],[2,0,1]) #,Q2,W,x,y,z,xF #,vT,vphi,vz,theta,vtheta #OLD
    pT_index, phi_index, eta_index, chi2_index, beta_index, status_index, charge_index, pid_index, theta_index,Q2_index ,W_index ,x_index ,y_index ,z_index ,xF_index = [i for i in range(datap.shape[-1])] #Q2_index, W_index, x_index, y_index, z_index, xF_index
    
    # Get mass array and copy so can write to it
    massp = np.moveaxis(ma.array([masses]),[0,1,2],[2,0,1])
    mass_index = 0
    
    # Mask out particles with nonvalid theta/eta values #NOTE: This just destroys the distributions currently...
    newmask = ma.mask_or(eta.mask,theta.mask)
    datap.mask = np.moveaxis(ma.array([newmask for el in range(0,datap.shape[-1])]),[0,1,2],[2,0,1])
    
    # Reassign particles with default chi2pid (9999.)
    chi2_max, chi2_default, chi2_replacement   = 10, 9999, 10
    datap[:,:,chi2_index] = ma.where(datap[:,:,chi2_index]>=chi2_default, chi2_replacement, datap[:,:,chi2_index])
    
    # Mask particles with chi2pid > max
    datap.mask = np.logical_or(datap.mask,np.moveaxis(
        ma.array([(np.absolute(datap[:,:,chi2_index])>chi2_max) for el in datap[0,0]])
        ,[0,1,2],[2,0,1]))
    
    # Mask particles with zero or negative beta
    beta_min, beta_default   = 0, 0.01
    datap.mask = np.logical_or(datap.mask,np.moveaxis(
        ma.array([(np.absolute(datap[:,:,beta_index])<=beta_min) for el in datap[0,0]])
        ,[0,1,2],[2,0,1]))
    
    # Preprocess status variable
    datap[:,:,status_index] /= 4000
        
    # Preprocess data
    for index in range(0,len(datap)):
        
        # Make sure percentage of lambda events is reasonable
        if (lambda_fraction>0):
            if targetp[index] == lambda_label:
                lambda_counter +=1 #NOTE: Remove events here if skim has inherently more lambda events
                if lambda_counter>(other_counter+lambda_counter)*lambda_fraction:
                    datap[index] = ma.masked_all(datap[index].shape)
                    targetp.mask[index] = True
                    massp[index] = ma.masked_all(massp[index].shape)
                    lambda_counter -=1
            else:
                other_counter +=1 #NOTE: Remove events here if skim has inherently less lambda events
#                 if other_counter>(other_counter+lambda_counter)*(1-lambda_fraction):
#                     datap[index] = ma.masked_all(datap[index].shape)
#                     massp[index] = ma.masked_all(massp[index].shape)
#                     targetp.mask[index] = True
#                     other_counter -=1
        
        # Make sure electron proton and pion are reconstructed in event
        if (strict):
            if not ((pids['e'] in pid[index])
                    and (pids['p'] in pid[index])
                    and (pids['pim'] in pid[index])):
                datap[index] = ma.masked_all(datap[index].shape)
                targetp.mask[index] = True
                massp[index] = ma.masked_all(massp[index].shape)
        
        # Get normalized differences from event mean or something similar
        mask   = ~datap[index,:,pT_index].mask
        datap[index,mask,chi2_index] /= chi2_max
        avePT  = np.mean(pT[index,mask]) if not isinstance(px[index,mask].sum(),ma.core.MaskedConstant) else 0
        datap[index,mask,pT_index]   -= avePT
        maxPT  = np.absolute(pT[index,mask]).max() if not isinstance(px[index,mask].sum(),ma.core.MaskedConstant) else 1
        datap[index,mask,pT_index]   /= (maxPT if (maxPT!=0) else 1)
        avePhi  = np.mean(phi[index,mask]) if not isinstance(px[index,mask].sum(),ma.core.MaskedConstant) else 0
        datap[index,mask,phi_index]  -= avePhi
        maxPhi  = np.absolute(phi[index,mask]).max() if not isinstance(px[index,mask].sum(),ma.core.MaskedConstant) else 1
        datap[index,mask,phi_index]  /= (maxPhi if maxPhi!=0 else 1)
        aveEta  = np.mean(eta[index,mask]) if not isinstance(px[index,mask].sum(),ma.core.MaskedConstant) else 0
        datap[index,mask,eta_index]  -= aveEta
        maxEta  = np.absolute(eta[index,mask]).max() if not isinstance(px[index,mask].sum(),ma.core.MaskedConstant) else 1
        datap[index,mask,eta_index]  /= (maxEta if maxEta!=0 else 1)
        aveTheta = np.mean(theta[index,mask]) if not isinstance(px[index,mask].sum(),ma.core.MaskedConstant) else 0
        datap[index,mask,theta_index]  -= aveTheta
        maxTheta = np.absolute(theta[index,mask]).max() if not isinstance(px[index,mask].sum(),ma.core.MaskedConstant) else 1
        datap[index,mask,theta_index]  /= (maxTheta if maxTheta!=0 else 1)
        aveBeta = np.mean(np.log(beta[index,mask])) if not isinstance(px[index,mask].sum(),ma.core.MaskedConstant) else 1
        maxBeta = np.absolute(beta[index,mask]).max() if not isinstance(px[index,mask].sum(),ma.core.MaskedConstant) else 1
        datap[index,mask,beta_index] = np.log(datap[index,mask,beta_index])
        datap[index,mask,beta_index] -= aveBeta
        datap[index,mask,beta_index] /= (maxBeta if maxBeta != 0 else 1)

    # Mask out particles with nonvalid beta values
    newmask = ma.mask_or(datap[:,:,beta_index].mask,datap[:,:,pT_index].mask)
    datap.mask = np.moveaxis(ma.array([newmask for el in range(0,9)]),[0,1,2],[2,0,1])

    # Replace PIDs if using
    if use_pids:
        replace_pids(datap,pid_index)
    
    # Remove completely masked events
    datap   = datap[~targetp.mask,:,:]#IMPORTANT: Make sure this is before you reassign targetp
    massp   = massp[~targetp.mask,:,:]#IMPORTANT: Make sure this is before you reassign targetp
    targetp = targetp[~targetp.mask]
    
    # Add to big arrays
    if counter == 0:
        data   = datap
        target = targetp
        mass   = massp
    else: 
        data   = ma.concatenate([data,datap],axis=0)
        target = ma.concatenate([target,targetp],axis=0)
        mass   = ma.concatenate([mass,massp],axis=0)
    
    # Print progress bar
    print('\r |',
          (counter*step_size*25)//max_events*'-','>',
          (25-(counter*step_size*25)//max_events)*' ',
          '| ',
          (counter*step_size*100)//max_events,
          '%',end='')
    counter +=1
    
    # Add data to ROOT file
    if (len(data)-1 - max_index) >= write_interval:
        
        # Update indices
        file_num += 1
        if max_index > len(data) - 1: max_index = len(data) - 1
        
        # Writing to ROOT not implemented for uproot 4.x
        try: os.mkdir(os.path.join(outdir,dataset_name))
        except FileExistsError: pass
        file = ur3.recreate(os.path.join(outdir,dataset_name,"events"+str(file_num)+".root"))
        file["tree"] = ur3.newtree({"label":  np.float32,
                                    "mass":   np.float32,
                                    "pT":     np.float32,
                                    "phi":    np.float32,
                                    "eta":    np.float32,
                                    "theta":  np.float32,
                                    "chi2":   np.float32,
                                    "beta":   np.float32,
                                    "status": np.float32,
                                    "charge": np.float32,
                                    "pid":    np.float32})
        # Add data to file
        file["tree"].extend({"label":target[min_index:max_index],
                         "mass":   mass[min_index:max_index,:,mass_index],
                         "pT":     data[min_index:max_index,:,pT_index],
                         "phi":    data[min_index:max_index,:,phi_index],
                         "theta":  data[min_index:max_index,:,theta_index],
                         "eta":    data[min_index:max_index,:,eta_index],
                         "chi2":   data[min_index:max_index,:,chi2_index],
                         "beta":   data[min_index:max_index,:,beta_index],
                         "status": data[min_index:max_index,:,status_index],
                         "charge": data[min_index:max_index,:,charge_index],
                         "pid":    data[min_index:max_index,:,pid_index]})
        
        # Write file -> IMPORTANT!
        file.close()
        
        # Update indices
        min_index = max_index
        max_index +=  step_size * write_interval
    
    # Break if max_events exceeded
    if counter * step_size >= max_events: break

# Get time taken (nicely formatted;)
time_taken = time.time() - start_time
end_date   = time.strftime('%X %x %Z')
if time_taken>=60**2: time_taken = str(round((time_taken)/60**2))+":"+str(round((time_taken)/60))+":"+str(round((time_taken)%60,2))
if time_taken>=60: time_taken = str(round((time_taken)/60))+":"+str(round((time_taken)%60,2))
else: time_taken = str(round((time_taken)%60,2))

# Check data and target shapes
print("\n Data shape   = ",np.shape(data))
print(" Target shape   = ",np.shape(target))
print(" Mass   shape   = ",np.shape(mass))
print(" Non-empty events: ",int(100*target.count()/target.shape[0]),"%")
print(" Lambda events   : ",(len(target[target>0])/len(target)*100)//1,"%")
print(" Time taken   = "+time_taken+" from "+start_date+" to "+end_date)
    
# Get separated mass distributions #TODO: separate into false/true positives/negatives
mass_sig_Y    = ma.array(mass[:,0,mass_index],mask=~(target == 1))
mass_bg_Y     = ma.array(mass[:,0,mass_index],mask=~(target == 0))

# Plot MC-Matched distributions
bins = 100
low_high = (1.1,1.13)
plt.title('Separated mass distribution MC-matched')
plt.hist(mass_sig_Y[~mass_sig_Y.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
plt.hist(mass_bg_Y[~mass_bg_Y.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
plt.legend(loc='upper left', frameon=False)
plt.ylabel('Counts')
plt.xlabel('Invariant mass (GeV)')
plt.show()

# Check data -> TODO: Remove outliers of (low, high)
def plot_data(array,title=None,xlabel="index",nbins=100,low=-1.1,high=1.1):
    if title != None:
        plt.title(title)
    plt.hist(array,nbins,(low,high))
    plt.xlabel(xlabel)
    plt.ylabel("counts")
    plt.show()

# Plot data distributions
plot_data(mass[:,:,mass_index][~mass[:,:,mass_index].mask].flatten(),xlabel="mass",low=1.09,high=1.14)
plot_data(data[:,:,pT_index][~data[:,:,pT_index].mask].flatten(),xlabel="pT")
plot_data(data[:,:,phi_index][~data[:,:,phi_index].mask].flatten(),xlabel="phi")
plot_data(data[:,:,theta_index][~data[:,:,theta_index].mask].flatten(),xlabel="theta")
plot_data(data[:,:,eta_index][~data[:,:,eta_index].mask].flatten(),xlabel="eta")
plot_data(data[:,:,chi2_index][~data[:,:,chi2_index].mask].flatten(),xlabel="chi2")
plot_data(data[:,:,beta_index][~data[:,:,beta_index].mask].flatten(),xlabel="beta")
plot_data(data[:,:,charge_index][~data[:,:,charge_index].mask].flatten(),xlabel="charge")
plot_data(data[:,:,pid_index][~data[:,:,pid_index].mask].flatten(),xlabel="pid")
plot_data(data[:,:,status_index][~data[:,:,status_index].mask].flatten(),xlabel="status")
plot_data(data[:,:,Q2_index][~data[:,:,Q2_index].mask].flatten(),xlabel="Q2")
plot_data(data[:,:,W_index][~data[:,:,W_index].mask].flatten(),xlabel="W")
plot_data(data[:,:,x_index][~data[:,:,x_index].mask].flatten(),xlabel="x")
plot_data(data[:,:,y_index][~data[:,:,y_index].mask].flatten(),xlabel="y")
plot_data(data[:,:,z_index][~data[:,:,z_index].mask].flatten(),xlabel="z")
plot_data(data[:,:,xF_index][~data[:,:,xF_index].mask].flatten(),xlabel="xF")

# Convert data events to DGL Graphs
mydataset = {"data":[], "target":[], "mass":[]}
counter = -1

for datap in data:

    # Update counter
    counter += 1

    # Get lists for unique index combinations
    nomask = datap[:,pT_index].count() #IMPORTANT: Just count along single data entry for all particles in event.
    if nomask<2: 
        print("*** WARNING *** Event",counter,"has just 1 particle.  Skipping.")
        continue
        
    mydataset["target"].append(target[counter])
    nums = [el for el in range(0,nomask)]
    l1, l2 = [], []
    for i in range(1,len(nums)):
        l1 += [nums[i-1] for x in range(0,(len(nums) - i))]
        l2 += nums[i:]

    # Get directional graph
    g = dgl.graph((l1,l2))

    # Get bidirectional graph
    bg = dgl.to_bidirected(g)

    # Get Individual data arrays #TODO -> This would probably be nicer if you used a dictionary.
    datap_pT  = datap[:,pT_index][~datap[:,pT_index].mask]
    datap_eta = datap[:,eta_index][~datap[:,pT_index].mask]
    datap_phi = datap[:,phi_index][~datap[:,pT_index].mask]
    datap_theta = datap[:,theta_index][~datap[:,pT_index].mask]
    datap_chi2 = datap[:,chi2_index][~datap[:,pT_index].mask]
    datap_status = datap[:,status_index][~datap[:,pT_index].mask]
    datap_charge = datap[:,charge_index][~datap[:,pT_index].mask]
    datap_beta = datap[:,beta_index][~datap[:,pT_index].mask]
    datap_pid = datap[:,pid_index][~datap[:,pT_index].mask]
    massp = mass[counter,:,mass_index]
    
    # Add node data to graph
    try:
        bg.ndata['data'] = torch.moveaxis(torch.tensor([datap_pT,datap_phi,datap_theta,datap_beta,datap_pid,datap_charge,datap_status,datap_chi2]),(0,1),(1,0))
    except Exception:
        print("*** WARNING SKIPPING EVENT: ",counter," ***")
        print(datap_pT.shape,datap_phi.shape,datap_eta.shape,datap_theta.shape)
        print(datap_chi2.shape,datap_status.shape,datap_charge.shape,datap_beta.shape,datap_pid.shape)
        continue
    
    # Get node difference tensors
    deltas = {"dpT":[],"deta":[],"dphi":[]}
    for j in range(0,len(bg.edges()[0])):

        # Get combination indices
        el1 = bg.edges()[0][j].item()
        el2 = bg.edges()[1][j].item()

        # Add data to arrays
        deltas["dpT"].append(abs(datap_pT[el1]-datap_pT[el2]))
        deltas["deta"].append(abs(datap_eta[el1]-datap_eta[el2]))
        deltas["dphi"].append(abs(datap_phi[el1]-datap_phi[el2]))

    # Add edge data to graph
    bg.edata['data'] = torch.tensor(np.moveaxis([deltas[key] for key in deltas.keys()],[0,1],[1,0]))
    
    # Add graph to dataset
    mydataset["data"].append(bg)
    mydataset["mass"].append(massp[0])#NOTE:  Doesn't take into account double match events...
        
# Sanity check
print(mydataset["data"][0].ndata)
print(mydataset["data"][0].edata)

# Shuffle dataset
indices = np.array([i for i in range(len(mydataset["data"]))])
np.random.shuffle(indices) #NOTE: In-place method
mydataset["data"]   = [mydataset["data"][i] for i in indices]
mydataset["target"] = [mydataset["target"][i] for i in indices]
mydataset["mass"]   = [mydataset["mass"][i] for i in indices]

# Split data into training and testing subsets
train = 0.75
train_index = int(train*len(mydataset["data"]))
mytraindataset = {"data":mydataset["data"][0:train_index],"target":mydataset["target"][0:train_index],"mass":mydataset["mass"][0:train_index]}
mytestdataset = {"data":mydataset["data"][train_index:],"target":mydataset["target"][train_index:],"mass":mydataset["mass"][train_index:]}
print(" Train dataset data shape   = ",len(mytraindataset["data"]))
print(" Train dataset target shape = ",len(mytraindataset["target"]))
print(" Train Lambda events        : ",(np.sum(mytraindataset["target"])/len(mytraindataset["target"])*100)//1,"%")
print(" Test dataset data shape    = ",len(mytestdataset["data"]))
print(" Test dataset target shape  = ",len(mytestdataset["target"]))
print(" Test Lambda events         : ",(np.sum(mytestdataset["target"])/len(mytestdataset["target"])*100)//1,"%")

# Add any additional identifiers to dataset name
# extra_name  = "_noStatus"
# extra_name  = "_noStatusPid"
# extra_name  = "_noStatusChi2"
# extra_name  = "_noStatusChi2PidBeta"
# extra_name = "_noChi2AddCharge"
# extra_name = "_noEtaNewChi2"
# extra_name = "no_EtaChi2"
# extra_name = "_noEtaOldChi2"
extra_name = "_noEtaOldChi2_noEdges"

# Load and save training data
train_dataset = LambdasDataset(dataset_name+extra_name+"_train",dataset=mytraindataset)
train_dataset.process()
train_dataset.save()

# Load and save testing data
test_dataset = LambdasDataset(dataset_name+extra_name+"_test",dataset=mytestdataset)
test_dataset.process()
test_dataset.save()
