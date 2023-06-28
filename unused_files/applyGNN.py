#------------------------------------------------------------#
#
#
#------------------------------------------------------------#

#------------------------------------------------------------#
# Imports
import numpy as np
from numpy import ma
import awkward as ak
import dgl
import torch
from models import GIN, MLP_SIGMOID, Concatenate
import models
import os

#------------------------------------------------------------#
# Exterior copied code

strict = True
use_pids = True
verbose = True

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

# pid_replacements = {
#     45:0.0,
#     22:0.0,#0.1,
#     111:0.2,
#     211:0.5,#0.25,
#     -211:-1,#0.3
#     311:0.35,
#     321:0.4,
#     -321:0.45,
#     2112:0.5,
#     -2212:0.6,
#     2212:1,#0.7,
#     -11:0.9,#0.8,
#     11:-0.5,
# }

pid_replacements = {
    22:0.0,
    11:-1.0,
    -11:1.0,
    2212:0.8,
    -2212:-0.8,
    2112:0.5,
    111:0.1,
    211:0.6,
    -211:-0.6,
    311:0.3,
    321:0.4,
    -321:-0.4,
    45:0.0
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

#------------------------------------------------------------#
# Functions

def load_model(nlayers,nmlp,hdim,dropout,learn_eps,npooling,gpooling,device,path,name):
    # Setup data and model
    nclasses, nfeatures, nfeatures_edge = 2, 7, 0 #get_graph_dataset_info(dataset=args.dataset, prefix=args.prefix)

    model = GIN(nlayers, nmlp, nfeatures,
            hdim, nclasses, dropout, learn_eps, npooling,
            gpooling).to(device)
    print("DEBUGGING: LOADING: ",os.path.join(path,name+'_weights'))#DEBUGGING
    model.load_state_dict(torch.load(os.path.join(path,name+'_weights'),map_location=device))

    return model

def load_modelDA(nlayers,nmlp,hdim,nmlp_head,hdim_head,dropout,learn_eps,npooling,gpooling,device,path,name):
    # Setup data and model
    nclasses, nfeatures, nfeatures_edge = 2, 7, 0 #get_graph_dataset_info(dataset=args.dataset, prefix=args.prefix)

    _model = GIN(nlayers, nmlp, nfeatures,
            hdim, hdim, dropout, learn_eps, npooling,
            gpooling).to(device)
    _classifier = MLP_SIGMOID(nmlp_head, hdim, hdim_head, nclasses).to(device)
    print("DEBUGGING: LOADING: ",os.path.join(path,name+'_model_weights'))#DEBUGGING
    print("DEBUGGING: LOADING: ",os.path.join(path,name+'_classifier_weights'))#DEBUGGING
    _model.load_state_dict(torch.load(os.path.join(path,name+'_model_weights'),map_location=device))
    _classifier.load_state_dict(torch.load(os.path.join(path,name+'_classifier_weights'),map_location=device))

    model = models.Concatenate([ _model, _classifier])

    return model

# My Function
def createGraph(batch):
    """
    Returns
    -------
    DGL Graph Object for given dictionary of event data.
    """
    # graph = dgl.Graph() #NOTE: INITIALIZE RETURN OBJECT

    # 
    targetD = 100
    #NOTE: BASIC LAMBDA LABELLING
    mc_label = ak.to_numpy(ak.pad_none(batch['MC::Lund_pid'],targetD,axis=1)) #NOTE: KEEP FOR MC
    targetp = ma.array([1 if pids['L0'] in arr else 0 for arr in mc_label]) #NOTE: KEEP FOR MC
    targetp.mask = [False for el in targetp]

    # mc_label = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_label'],targetD,axis=1)) #NOTE: KEEP FOR MC
    # mc_pa_pid = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_pid_pa_MC'],targetD,axis=1)) #NOTE: KEEP FOR MC

    # targetp = ma.array([1 if 1 in arr else 0 for arr in mc_label]) #NOTE: KEEP FOR MC
    # targetp.mask = [False for el in targetp]
    # my_pid=3122#NOTE: LABELLING NOT SKIM! DO NOT CHANGE
    # targetp_pa_pid = ma.array([1 if my_pid in arr else 0 for arr in mc_pa_pid]) #NOTE: KEEP FOR MC
    # targetp = ma.minimum(targetp,targetp_pa_pid)
    # targetp.mask = [False for el in targetp]
    # skim_pid=2114#NOTE: SKIM NOT LABEL! CHANGE
    # targetp_pa_pid__ = ma.array([1 if (skim_pid not in arr and my_pid not in arr) else 0 for arr in mc_pa_pid]) #NOTE: KEEP FOR MC #NOTE: USE THIS ONE TO FILTER FOR PARENT PID SKIMS
    # #COMMENTED OUT FOR BACKGROUND FRACTION ESTIMATION

    px      = ak.to_numpy(ak.pad_none(batch['REC::Particle_px'],targetD,axis=1))
    py      = ak.to_numpy(ak.pad_none(batch['REC::Particle_py'],targetD,axis=1))
    pz      = ak.to_numpy(ak.pad_none(batch['REC::Particle_pz'],targetD,axis=1))
    pid     = ak.to_numpy(ak.pad_none(batch['REC::Particle_pid'],targetD,axis=1))
    chi2    = ak.to_numpy(ak.pad_none(batch['REC::Particle_chi2pid'],targetD,axis=1))
    status  = ak.to_numpy(ak.pad_none(batch['REC::Particle_status'],targetD,axis=1))
    charge  = ak.to_numpy(ak.pad_none(batch['REC::Particle_charge'],targetD,axis=1))
    beta    = ak.to_numpy(ak.pad_none(batch['REC::Particle_beta'],targetD,axis=1))
    
    # Compute some more useful arrays # Could also add theta and vertices -> phi and vT vz
    mask      = ~px[:,:].mask
    pT        = ma.array(np.zeros(np.shape(px)),mask=~mask)
    pT[mask]  = np.add(px[mask]**2,py[mask]**2)
    p2        = ma.array(np.zeros(np.shape(px)),mask=~mask)
    p2[mask]  = np.add(pT[mask]**2,pz[mask]**2)
#     ###pT[mask]  = np.sqrt(p2[mask])#NOTE: NEW!!! 7/27/22 CHanged pT to just p ... just keep as is...
    theta       = ma.array(np.zeros(np.shape(px)),mask=~mask)
    theta[mask] = np.divide(pT[mask],pz[mask])
    phi       = ma.array(np.zeros(np.shape(px)),mask=~mask)
    phi[mask] = np.arctan(py[mask],px[mask])
    eta       = ma.array(np.zeros(np.shape(px)),mask=~mask)
    eta[mask] = np.arctanh(np.divide(pz[mask],np.sqrt(p2[mask])))
    
    # Get data array and move eventwise axis to be first
    datap   = np.moveaxis(ma.array([pT,phi,eta,chi2,beta,status,charge,pid,theta]),[0,1,2],[2,0,1]) #,Q2,W,x,y,z,xF #,vT,vphi,vz,theta,vtheta #OLD
    pT_index, phi_index, eta_index, chi2_index, beta_index, status_index, charge_index, pid_index, theta_index = [i for i in range(datap.shape[-1])] #Q2_index, W_index, x_index, y_index, z_index, xF_index
    
    # Mask out particles with nonvalid theta/eta values #NOTE: This just destroys the distributions currently...
    newmask = np.logical_or(eta.mask,theta.mask)
    datap.mask = np.moveaxis(ma.array([newmask for el in range(0,datap.shape[-1])]),[0,1,2],[2,0,1])
    
    # Reassign particles with default chi2pid (9999.)
    chi2_max, chi2_default, chi2_replacement   = 10, 9999, 10 #NOTE: DEBUGGING: USED TO BE >=chi2_default below!
    datap[:,:,chi2_index] = ma.where(datap[:,:,chi2_index]>=chi2_default, chi2_replacement, datap[:,:,chi2_index])
    
    # Mask particles with chi2pid > max
    datap.mask = np.logical_or(datap.mask,np.moveaxis(
        ma.array([(np.absolute(datap[:,:,chi2_index])>chi2_max) for el in datap[0,0]])
        ,[0,1,2],[2,0,1]))
    
    # Preprocess chi2 variable
    datap[:,:,chi2_index] /= chi2_max #NOTE: IMPORTANT!  NEEDS TO BE AFTER REPLACEMENT OF MAX ABOVE!
    
    # Mask particles with zero or negative beta
    beta_min, beta_default   = 0, 0.01
    datap.mask = np.logical_or(datap.mask,np.moveaxis(
        ma.array([(np.absolute(datap[:,:,beta_index])<=beta_min) for el in datap[0,0]])
        ,[0,1,2],[2,0,1]))
    
    # Preprocess status variable
    datap[:,:,status_index] /= 5000
       
    # Preprocess data
    for index in range(0,len(datap)):
        
        # Make sure electron proton and pion are reconstructed in event
        if (strict):
            if not ((pids['e'] in pid[index])
                    and (pids['p'] in pid[index])
                    and (pids['pim'] in pid[index])):
                datap[index] = ma.masked_all(datap[index].shape)
                targetp.mask[index] = True
                # massp[index] = ma.masked_all(massp[index].shape)
        
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
    targetp = targetp[~targetp.mask]
    
    # Add to big arrays
    data   = datap #ma.concatenate([data,datap],axis=0)
    target = targetp #ma.concatenate([target,targetp],axis=0)

    # Convert data events to DGL Graphs

    print(np.shape(data))#DEBUGGING
    print(np.shape(datap))#DEBUGGING

    # Get lists for unique index combinations
    nomask = datap[:,pT_index].count() #IMPORTANT: Just count along single data entry for all particles in event.
    if nomask<2: 
        print("*** WARNING *** Event has just 1 particle.  Skipping.")
        print(datap[:,pid_index])
        return dgl.graph(([],[]))
        
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
    # massp = mass[counter,:,mass_index]
    
    # Add node data to graph
#     try:
    bg.ndata['data'] = torch.moveaxis(torch.tensor(ma.array([datap_pT,datap_phi,datap_theta,datap_beta,datap_chi2,datap_pid,datap_status])),(0,1),(1,0))
    if verbose: print("bg.ndata['data'] = ",bg.ndata['data'])#DEBUGGING
#         print(ma.array([datap_pT,datap_phi,datap_theta,datap_beta,datap_pid,datap_charge,datap_status,datap_chi2]))
#         break#DEBUGGING
#     except Exception:
#         print("*** WARNING SKIPPING EVENT: ",counter," ***")
#         print(datap_pT.shape,datap_phi.shape,datap_eta.shape,datap_theta.shape)
#         print(datap_chi2.shape,datap_status.shape,datap_charge.shape,datap_beta.shape,datap_pid.shape)
#         return dgl.graph(([],[]))
    
#     # Get node difference tensors
#     deltas = {"dpT":[],"deta":[],"dphi":[]}
#     for j in range(0,len(bg.edges()[0])):

#         # Get combination indices
#         el1 = bg.edges()[0][j].item()
#         el2 = bg.edges()[1][j].item()

#         # Add data to arrays
#         deltas["dpx"].append(abs(datap_px[el1]-datap_px[el2]))
#         deltas["dpy"].append(abs(datap_py[el1]-datap_py[el2]))
#         deltas["dpz"].append(abs(datap_pz[el1]-datap_pz[el2]))

#     # Add edge data to graph
#     bg.edata['data'] = torch.tensor(np.moveaxis([deltas[key] for key in deltas.keys()],[0,1],[1,0]))

    graph = bg
    return graph

def apply(model,graph):
    """
    Returns
    -------
    Label prediction of given model on graph
    """

    # Apply model to input graph
    prediction = model(graph)
    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)

    # Copy arrays back to CPU
    probs_Y  = probs_Y.cpu()
    argmax_Y = argmax_Y.cpu()

    return argmax_Y, probs_Y

#------------------------------------------------------------#
# Classes

# My class


