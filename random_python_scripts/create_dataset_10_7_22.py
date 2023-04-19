#----------------------------------------------------------------------#
# Example notebook for reading HIPO files into DGL graphs
# Authors: M. McEneaney (2022, Duke University)
#----------------------------------------------------------------------#

import numpy as np
import numpy.ma as ma
import awkward as ak
import hipopy.hipopy as hippy

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
import math, datetime, os, time, sys

# sys.path.append('.')
# from utils import GraphDataset

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

# Loop variables
use_pids = True
step_size = 10
write_interval = 100 # How many iterations to wait before writing array to root
min_index = 0
max_index = step_size * write_interval
file_num = 0
write_counter = 0
max_events = 550000#10**6 # +16000# * 10**4 + 1
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
num_workers = 2**5 # 2**3 is best on 4 or 8 cores, 2**5 is really slow.

## Reading a chain of files
data   = []
target = []
mass   = []
counter = 0
print("#----------------------------------------------------------------------#")
#NOTE: LambdaTrain_data_jobs_ppim_rga_nSidis_outbending_7_28_22 is all .root but hipo files actually... will switch at some point
# filenames = ['/volatile/clas12/users/mfmce/TRAIN_SKIM_SMALL_rga_fall2018_nobg_jobs_ppim_8_22_22/*.hipo']
filenames = ['/volatile/clas12/users/mfmce/TRAIN_SKIM_SMALL_cache_clas12_rga_fall2018_bg50nA_jobs_ppim_8_22_22/*.hipo']
# filenames = ['/volatile/clas12/users/mfmce/LambdaTrain_data_jobs_ppim_rga_nSidis_outbending_7_28_22/*.hipo'] #NOTE: NEW DATA 7/28/22
# filenames = ['/volatile/clas12/users/mfmce/TRAIN_SKIM_SMALL_cache_clas12_rga_fall2018_bg50nA_jobs_ppim_7_28_22/__labels__*.hipo']#NOTE: NEW 7/28/22 Just require Lambda->ppi- in MC Truth
# filenames = ['/volatile/clas12/users/mfmce/TRAIN_SKIM_SMALL_cache_clas12_rga_fall2018_bg50nA_jobs_ppim_7_21_22/__labels__*.hipo']#NOTE: OLD MC
# filenames = ['/volatile/clas12/users/mfmce/LambdaTrain_cache_clas12_jobs_rga_bg50nA_7_7_22/*.hipo']#NOTE: OLD MC
# filenames = ['/volatile/clas12/users/mfmce/Lambda_train_jobs_outbending_4_28_22/*.hipo']#NOTE: Make sure to specify the full path or relative path from directory in which you call this script.
# banks = ["REC::Particle","REC::Kinematics","MC::Lund"] #NOTE: MC #NOTE: OLD AS OF 7/20/22
banks = ["REC::Particle","MC::Lund","MC::TruthMatching_SMALL"] #NOTE: MC
# banks = ["REC::Particle","REC::Kinematics"] #NOTE: DATA
counter = -1
step = 100

# Loop through batches of step # events in the chain.
start_date = time.strftime('%X')
start_time = time.time()
nth = 0
start = nth * max_events / step_size
for batch in hippy.iterate(filenames,banks=banks,step=step): # If you don't specify banks, ALL banks will be read.
    counter +=1

    # Break if max_events exceeded
    if counter * step_size - start*step_size > max_events: break
    if counter<start: continue #NOTE: NEW FOR BATCHING ENTIRE PROCESS 7/26/22
    if counter == start: print("STARTING at counter = ",counter)#DEBUGGING
        
#     # Get target array
# #     mc_pid      = ak.to_numpy(ak.pad_none(batch['MC::Lund_pid'],targetD,axis=1))
# #     mc_par      = ak.to_numpy(ak.pad_none(batch['MC::Lund_parent'],targetD,axis=1))
# #     mc_dau      = ak.to_numpy(ak.pad_none(batch['MC::Lund_daughter'],targetD,axis=1))
# #     mc_px       = ak.to_numpy(ak.pad_none(batch['MC::Lund_px'],targetD,axis=1))
# #     mc_py       = ak.to_numpy(ak.pad_none(batch['MC::Lund_py'],targetD,axis=1))
# #     mc_pz       = ak.to_numpy(ak.pad_none(batch['MC::Lund_pz'],targetD,axis=1))
    
#     # NEW
# #     mc_idx_pa = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_idx_pa_MC'],targetD,axis=1))
# #     mc_idx_pa_pim = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_idx_pa_pim_MC'],targetD,axis=1))
# #     mc_pid_pa = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_pid_pa_MC'],targetD,axis=1))

    mc_label = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_label'],targetD,axis=1)) #NOTE: KEEP FOR MC
    mc_pa_pid = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_pid_pa_MC'],targetD,axis=1)) #NOTE: KEEP FOR MC
    #----------#
    mc_idx_p = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_idx_p_MC'],targetD,axis=1)) #NOTE: KEEP FOR MC
    mc_idx_pim = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_idx_pim_MC'],targetD,axis=1)) #NOTE: KEEP FOR MC
    mc_idx_pa_p = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_idx_pa_MC'],targetD,axis=1)) #NOTE: KEEP FOR MC
    mc_idx_pa_pim = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_idx_pa_pim_MC'],targetD,axis=1)) #NOTE: KEEP FOR MC
    mc_pid_ppa = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_pid_ppa_MC'],targetD,axis=1)) #NOTE: KEEP FOR MC
    mc_idx_ppa = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_idx_ppa_MC'],targetD,axis=1)) #NOTE: KEEP FOR MC
    #----------#
# #     print(mc_label)#DEBUGGING
# #     break#DEBUGGING

#     print("batch['MC::TruthMatching_SMALL_pid_pa_MC'] = ")#DEBUGGING
#     print(batch['MC::TruthMatching_SMALL_pid_pa_MC'])#DEBUGGING
#     print("np.shape(batch['MC::TruthMatching_SMALL_pid_pa_MC']) = ",np.shape(batch['MC::TruthMatching_SMALL_pid_pa_MC']))#DEBUGGING
    
    targetp = ma.array([1 if 1 in arr else 0 for arr in mc_label]) #NOTE: KEEP FOR MC
    targetp.mask = [False for el in targetp]
    my_pid=3122#NOTE: LABELLING NOT SKIM! DO NOT CHANGE
    targetp_pa_pid = ma.array([1 if my_pid in arr else 0 for arr in mc_pa_pid]) #NOTE: KEEP FOR MC
    targetp = ma.minimum(targetp,targetp_pa_pid)
    targetp.mask = [False for el in targetp]
    skim_pid=2114#NOTE: SKIM NOT LABEL! CHANGE
    targetp_pa_pid__ = ma.array([1 if (skim_pid not in arr and my_pid not in arr) else 0 for arr in mc_pa_pid]) #NOTE: KEEP FOR MC #NOTE: USE THIS ONE TO FILTER FOR PARENT PID SKIMS
    #COMMENTED OUT FOR BACKGROUND FRACTION ESTIMATION
# #     print("DEBUGGING0: np.shape(targetp) = ",np.shape(targetp))#DEBUGGING
    
# #     targetp = ma.array([1 if mc_idx_pa[idx]==mc_idx_pa_pim[idx] and mc_pid_pa[idx]==3122 else 0 for idx in range(len(mc_idx_pa))])
    
# #     print(ma.count(mc_pid))
# #     print(ma.count(mc_par))
# #     print(ma.count(mc_dau))
# #     print(ma.count(mc_px))
# #     print(ma.count(mc_py))
# #     print(ma.count(mc_pz))

    
#NOTE: KEEP BELOW FOR DATA LABELLING
#     mc_pid = None
#     try:
#         mc_pid      = ak.to_numpy(ak.pad_none(batch['REC::Kinematics_mass'],targetD,axis=1))
#     except KeyError:
#         continue
# #     targetp     = ma.array([lambda_label if pids["L0"] in arr else 0 for arr in mc_pid],mask=[False for arr in mc_pid])
#     targetp     = ma.array([0 for arr in mc_pid],mask=[False for arr in mc_pid])
    
    # Get data arrays
#     run     = ak.to_numpy(batch['RUN::Config_event'])

#NOTE: UNCOMMENT BELOW FOR MC
    masses  = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_Mh'],targetD,axis=1))
    Q2      = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_Q2'],targetD,axis=1))
    W       = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_W'],targetD,axis=1))
    x       = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_x'],targetD,axis=1))
    y       = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_y'],targetD,axis=1))
    z       = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_z_pa'],targetD,axis=1))
    xF      = ak.to_numpy(ak.pad_none(batch['MC::TruthMatching_SMALL_xF_pa'],targetD,axis=1))
    
# #NOTE: UNCOMMENT BELOW FOR DATA
#     masses  = ak.to_numpy(ak.pad_none(batch['REC::Kinematics_mass'],targetD,axis=1))
#     Q2      = ak.to_numpy(ak.pad_none(batch['REC::Kinematics_Q2'],targetD,axis=1))
#     W       = ak.to_numpy(ak.pad_none(batch['REC::Kinematics_W'],targetD,axis=1))
#     x       = ak.to_numpy(ak.pad_none(batch['REC::Kinematics_x'],targetD,axis=1))
#     y       = ak.to_numpy(ak.pad_none(batch['REC::Kinematics_y'],targetD,axis=1))
#     z       = ak.to_numpy(ak.pad_none(batch['REC::Kinematics_z'],targetD,axis=1))
#     xF      = ak.to_numpy(ak.pad_none(batch['REC::Kinematics_xF'],targetD,axis=1))
    
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
    
#     #DEBUGGING BEGIN
#     print("DEBUGGING: type(batch['MC::TruthMatching_SMALL_label']) = ",type(batch['MC::TruthMatching_SMALL_label']))#DEBUGGING
#     print("DEBUGGING: np.shape(batch['MC::TruthMatching_SMALL_label']) = ",np.shape(batch['MC::TruthMatching_SMALL_label']))#DEBUGGING
#     print("DEBUGGING: type(batch['MC::TruthMatching_SMALL_label'][0]) = ",type(batch['MC::TruthMatching_SMALL_label'][0]))#DEBUGGING
#     print("DEBUGGING: np.shape(batch['MC::TruthMatching_SMALL_label'][0]) = ",np.shape(batch['MC::TruthMatching_SMALL_label'][0]))#DEBUGGING
#     print("DEBUGGING: batch['MC::TruthMatching_SMALL_label'][0] = ",batch['MC::TruthMatching_SMALL_label'][0])#DEBUGGING
    
    
#     print("DEBUGGING: type(batch['REC::Particle_px']) = ",type(batch['REC::Particle_px']))#DEBUGGING
#     print("DEBUGGING: np.shape(batch['REC::Particle_px']) = ",np.shape(batch['REC::Particle_px']))#DEBUGGING
#     print("DEBUGGING: type(batch['REC::Particle_px'][0]) = ",type(batch['REC::Particle_px'][0]))#DEBUGGING
#     print("DEBUGGING: np.shape(batch['REC::Particle_px'][0]) = ",np.shape(batch['REC::Particle_px'][0]))#DEBUGGING
#     print("DEBUGGING: batch['REC::Particle_px'][0] = ",batch['REC::Particle_px'][0])#DEBUGGING
    
#     print("DEBUGGING: type(mc_label) = ",type(mc_label))#DEBUGGING
#     print("DEBUGGING: np.shape(mc_label) = ",np.shape(mc_label))#DEBUGGING
#     print("DEBUGGING: type(mc_label[0]) = ",type(mc_label[0]))#DEBUGGING
#     print("DEBUGGING: np.shape(mc_label[0]) = ",np.shape(mc_label[0]))#DEBUGGING
#     print("DEBUGGING: mc_label[0] = ",mc_label[0])#DEBUGGING
    
    
#     print("DEBUGGING: type(px) = ",type(px))#DEBUGGING
#     print("DEBUGGING: np.shape(px) = ",np.shape(px))#DEBUGGING
#     print("DEBUGGING: type(px[0]) = ",type(px[0]))#DEBUGGING
#     print("DEBUGGING: np.shape(px[0]) = ",np.shape(px[0]))#DEBUGGING
#     print("DEBUGGING: px[0] = ",px[0])#DEBUGGING
#     break#DEBUGGING END
    
#     # Compute some more useful MC arrays
#     mc_mask      = ~mc_px[:,:].mask
#     mc_pT        = ma.array(np.zeros(np.shape(mc_px)),mask=~mc_mask)
#     mc_pT[mc_mask]  = np.add(mc_px[mc_mask]**2,mc_py[mc_mask]**2)
#     mc_theta       = ma.array(np.zeros(np.shape(mc_px)),mask=~mc_mask)
#     mc_theta[mc_mask] = np.divide(mc_pT[mc_mask],mc_pz[mc_mask])
#     mc_phi       = ma.array(np.zeros(np.shape(mc_px)),mask=~mc_mask)
#     mc_phi[mc_mask] = np.arctan(mc_py[mc_mask],mc_px[mc_mask])
    
#     print(np.shape(mc_pid))
#     start_idx, stop_idx = 0, 1
#     print(mc_pid[start_idx:stop_idx])
# #     print(mc_theta[start_idx:stop_idx])
#     print(mc_idx_pa[start_idx:stop_idx])
#     print(mc_idx_pa_pim[start_idx:stop_idx])
#     print(mc_pid_pa[start_idx:stop_idx])
#     i_, j_, k_, l_ = -1, -1, -1, -1
#     for event in mc_pid:
#         i_ += 1
# #         print("EVENT = ",event)
#         for pid in event:
#             j_ += 1
#             if pid==3122:
#                 for pid2 in event[j_:]:
#                     k_ += 1
                    
#     break #DEBUGGING
    
    # Get data array and move eventwise axis to be first
    datap   = np.moveaxis(ma.array([pT,phi,eta,chi2,beta,status,charge,pid,theta]),[0,1,2],[2,0,1]) #,Q2,W,x,y,z,xF #,vT,vphi,vz,theta,vtheta #OLD
    pT_index, phi_index, eta_index, chi2_index, beta_index, status_index, charge_index, pid_index, theta_index = [i for i in range(datap.shape[-1])] #Q2_index, W_index, x_index, y_index, z_index, xF_index
#     print("DEBUGGING1: np.shape(datap) = ",np.shape(datap))#DEBUGGING
#     print("DEBUGGING1: np.shape(pT)    = ",np.shape(pT))#DEBUGGING
# #     break#DEBUGGING
    # Get mass array and copy so can write to it
    massp = np.moveaxis(ma.array([masses,Q2,W,x,y,z,xF,mc_pa_pid,mc_pid_ppa]),[0,1,2],[2,0,1])#NOTE: UPDATED TO ADD PPA_PID 9/19/22
    mass_index, Q2_index, W_index, x_index, y_index, z_index, xF_index, mc_pid_pa_index, mc_pid_ppa_index = [i for i in range(massp.shape[-1])]
    mass_index = 0
    
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
    
#     print("DEBUGGING2: np.shape(datap) = ",np.shape(datap))#DEBUGGING
#     print("DEBUGGING2: np.shape(pT)    = ",np.shape(pT))#DEBUGGING
# #     break#DEBUGGING
       
    # Preprocess data
    for index in range(0,len(datap)):
#         print("DEBUGGING: INDEX = ",index)#DEBUGGING
        
#         if targetp_pa_pid__[index]!=1: #NOTE: COMMENTED OUT 10/7/22
#             datap[index] = ma.masked_all(datap[index].shape)
#             targetp.mask[index] = True
#             massp[index] = ma.masked_all(massp[index].shape)
#             continue
        
#         # Make sure percentage of lambda events is reasonable
#         if (lambda_fraction>0):
#             if targetp[index] == lambda_label:
#                 lambda_counter +=1 #NOTE: Remove events here if skim has inherently more lambda events
#                 if lambda_counter>(other_counter+lambda_counter)*lambda_fraction:
#                     datap[index] = ma.masked_all(datap[index].shape)
#                     targetp.mask[index] = True
#                     massp[index] = ma.masked_all(massp[index].shape)
#                     lambda_counter -=1
#             else:
#                 other_counter +=1 #NOTE: Remove events here if skim has inherently less lambda events
# #                 if other_counter>(other_counter+lambda_counter)*(1-lambda_fraction):
# #                     datap[index] = ma.masked_all(datap[index].shape)
# #                     massp[index] = ma.masked_all(massp[index].shape)
# #                     targetp.mask[index] = True
# #                     other_counter -=1
        
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
        
#     print("DEBUGGING3: np.shape(datap) = ",np.shape(datap))#DEBUGGING
#     print("DEBUGGING3: np.shape(pT)    = ",np.shape(pT))#DEBUGGING
# #     break#DEBUGGING

    # Mask out particles with nonvalid beta values
    newmask = ma.mask_or(datap[:,:,beta_index].mask,datap[:,:,pT_index].mask)
    datap.mask = np.moveaxis(ma.array([newmask for el in range(0,9)]),[0,1,2],[2,0,1])
    
#     print("DEBUGGING4: np.shape(datap) = ",np.shape(datap))#DEBUGGING
#     print("DEBUGGING4: np.shape(pT)    = ",np.shape(pT))#DEBUGGING
# #     break#DEBUGGING

    # Replace PIDs if using
    if use_pids:
        replace_pids(datap,pid_index)
        
#     print("DEBUGGING5: np.shape(datap) = ",np.shape(datap))#DEBUGGING
#     print("DEBUGGING5: np.shape(pT)    = ",np.shape(pT))#DEBUGGING
# #     break#DEBUGGING
    
    # Remove completely masked events
    datap   = datap[~targetp.mask,:,:]#IMPORTANT: Make sure this is before you reassign targetp
    massp   = massp[~targetp.mask,:,:]#IMPORTANT: Make sure this is before you reassign targetp
    targetp = targetp[~targetp.mask]

#     print("DEBUGGING: len(datap) = ",len(datap))#DEBUGGING
#     print("DEBUGGING: len(targetp) = ",len(targetp))#DEBUGGING
#     print("DEBUGGING: len(massp) = ",len(massp))#DEBUGGING
    
#     print("DEBUGGING6: np.shape(datap) = ",np.shape(datap))#DEBUGGING
#     print("DEBUGGING6: np.shape(pT)    = ",np.shape(pT))#DEBUGGING
#     break#DEBUGGING
    
    # Add to big arrays
    if counter == 0 or len(data)==0:
        data   = datap
        target = targetp
        mass   = massp
    else: 
#         print("DEBUGGING: BEFORE")
#         print("DEBUGGING: np.shape(data) = ",np.shape(data))#DEBUGGING
#         print("DEBUGGING: np.shape(datap) = ",np.shape(datap))#DEBUGGING
#         print("DEBUGGING: np.shape(target) = ",np.shape(target))#DEBUGGING
#         print("DEBUGGING: np.shape(targetp) = ",np.shape(targetp))#DEBUGGING
        data   = ma.concatenate([data,datap],axis=0)
        target = ma.concatenate([target,targetp],axis=0)
        mass   = ma.concatenate([mass,massp],axis=0)
#         print("DEBUGGING: AFTER")
#         print("DEBUGGING: np.shape(data) = ",np.shape(data))#DEBUGGING
#         print("DEBUGGING: np.shape(datap) = ",np.shape(datap))#DEBUGGING
#         print("DEBUGGING: np.shape(target) = ",np.shape(target))#DEBUGGING
#         print("DEBUGGING: np.shape(targetp) = ",np.shape(targetp))#DEBUGGING
    
    # Print progress bar
    print('\r |',
          int(max(counter-start,0)*step_size*25)//max_events*'-','>',
          int(25-(max(counter-start,0)*step_size*25)//max_events)*' ',
          '| ',
          int(max(counter-start,0)*step_size*100)//max_events,
          '% ',
          int(max(counter-start,0)*step),
          'events',
          end='')
    
#         print("DEBUGGING: np.shape(batch[\"REC::Particle_px\"]) = ",np.shape(batch["REC::Particle_px"]))#DEBUGGING
#         print("DEBUGGING: np.shape(batch[\"MC::TruthMatching_SMALL_label\"]) = ",np.shape(batch["MC::TruthMatching_SMALL_label"]))#DEBUGGING
#         print("DEBUGGING: np.shape(batch[\"REC::Particle_px\"][0]) = ",np.shape(batch["REC::Particle_px"][0]))#DEBUGGING
#         print("DEBUGGING: np.shape(batch[\"MC::TruthMatching_SMALL_label\"][0]) = ",np.shape(batch["MC::TruthMatching_SMALL_label"][0]))#DEBUGGING
#         break

# # Get time taken (nicely formatted;)
# time_taken = time.time() - start_time
# end_date   = time.strftime('%X %x %Z')
# if time_taken>=60**2: time_taken = str(round((time_taken)/60**2))+":"+str(round((time_taken)/60))+":"+str(round((time_taken)%60,2))
# if time_taken>=60: time_taken = str(round((time_taken)/60))+":"+str(round((time_taken)%60,2))
# else: time_taken = str(round((time_taken)%60,2))

# Check data and target shapes
print("\n Data shape   = ",np.shape(data))
print(" Target shape   = ",np.shape(target))
print(" Mass   shape   = ",np.shape(mass))
print(" Non-empty events: ",int(100*ma.count(target)/np.shape(target)[0]),"%")
print(" Lambda events   : ",(len(target[target>0])/len(target)*100)//1,"%")
# print(" Time taken   = "+time_taken+" from "+start_date+" to "+end_date)
    
# Get separated mass distributions #TODO: separate into false/true positives/negatives
mass_sig_Y    = ma.array(mass[:,0,mass_index],mask=~(target == 1))
mass_bg_Y     = ma.array(mass[:,0,mass_index],mask=~(target == 0))

# Plot MC-Matched distributions
bins = 100
low_high = (1.08,1.24)
plt.title('Separated mass distribution MC-matched')
plt.hist(mass_sig_Y[~mass_sig_Y.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
plt.hist(mass_bg_Y[~mass_bg_Y.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
plt.legend(loc='upper left', frameon=False)
plt.ylabel('Counts')
plt.xlabel('Invariant mass (GeV)')
plt.show()

print(len(data))
print(len(mass))
print(len(target))

# data = data[0:100000]
# target = target[0:100000]
# mass = mass[0:100000]
# Check data -> TODO: Remove outliers of (low, high)

# data__ = data #UNDO
# target__ = target
# mass__ = mass
# data = data_
# target = target_
# mass = mass_

# data_ = data #REDO
# mass_ = mass
# target_ = target
# data = data__
# target = target__
# mass = mass__

def plot_data(array,title=None,xlabel="index",nbins=100,low=-1.1,high=1.1):
    f = plt.figure()
    if title != None:
        plt.title(title)
    plt.hist(array,nbins,(low,high))
    plt.xlabel(xlabel)
    plt.ylabel("counts")
    plt.show()
    f.savefig(xlabel+'_MC_bg50nA_10_7_22_Combo_skim.png')

# Plot data distributions
plot_data(mass[:,:,mass_index][~mass[:,:,mass_index].mask].flatten(),xlabel="mass",low=1.08,high=1.24)
plot_data(data[:,:,pT_index][~data[:,:,pT_index].mask].flatten(),xlabel="pT")
plot_data(data[:,:,phi_index][~data[:,:,phi_index].mask].flatten(),xlabel="phi")
plot_data(data[:,:,theta_index][~data[:,:,theta_index].mask].flatten(),xlabel="theta")
plot_data(data[:,:,eta_index][~data[:,:,eta_index].mask].flatten(),xlabel="eta")
plot_data(data[:,:,chi2_index][~data[:,:,chi2_index].mask].flatten(),xlabel="chi2")
plot_data(data[:,:,beta_index][~data[:,:,beta_index].mask].flatten(),xlabel="beta")
plot_data(data[:,:,charge_index][~data[:,:,charge_index].mask].flatten(),xlabel="charge")
plot_data(data[:,:,pid_index][~data[:,:,pid_index].mask].flatten(),xlabel="pid")
plot_data(data[:,:,status_index][~data[:,:,status_index].mask].flatten(),xlabel="status")
plot_data(mass[:,:,Q2_index][~mass[:,:,Q2_index].mask].flatten(),xlabel="Q2",low=0.0,high=8.0)
plot_data(mass[:,:,W_index][~mass[:,:,W_index].mask].flatten(),xlabel="W",low=1.0,high=5.0)
plot_data(mass[:,:,x_index][~mass[:,:,x_index].mask].flatten(),xlabel="x",low=0.0,high=1.0)
plot_data(mass[:,:,y_index][~mass[:,:,y_index].mask].flatten(),xlabel="y",low=0.0,high=1.0)
plot_data(mass[:,:,z_index][~mass[:,:,z_index].mask].flatten(),xlabel="z",low=0.0,high=1.0)
plot_data(mass[:,:,xF_index][~mass[:,:,xF_index].mask].flatten(),xlabel="xF",low=-1.0,high=1.0)
# x_ = 0
# data_ = data
# mass_ = mass
# target_ = target
# # data = data_
# # mass = mass_
# # target = target_
# for datap in data:#DEBUGGING
#     print("datap[:,pT_index] = ",datap[:,pT_index])#DEBUGGING
#     print("datap[:,theta_index] = ",datap[:,theta_index])#DEBUGGING
#     print("datap[:,eta_index] = ",datap[:,eta_index])#DEBUGGING
#     print("datap[:,phi_index] = ",datap[:,phi_index])#DEBUGGING
#     print("datap[:,beta_index] = ",datap[:,beta_index])#DEBUGGING
#     print("datap[:,chi2_index] = ",datap[:,chi2_index])#DEBUGGING
#     print("datap[:,pid_index] = ",datap[:,pid_index])#DEBUGGING
#     print("datap[:,charge_index] = ",datap[:,charge_index])#DEBUGGING
#     print("datap[:,status_index] = ",datap[:,status_index])#DEBUGGING
#     print("--------------------------------------")#DEBUGGING
#     if x_ > 5: break#DEBUGGING
#     else: x_ += 1

#----- Now split so that you have equal unbiased samples for each class (sig/bg) -----#

# Count unique labels and get # of the least frequent
un_labels = np.unique(target) # Get number of unique labels
n = np.min(np.unique(target,return_counts=True)[1]) # Get label with least number of counts

# Get lists by label
helper        = [np.where(target==label)[0] for label in un_labels] # Get addresses for each label 
target_helper = [np.array(target)[h][0:n] for h in helper] # Get target lists by label
data_helper   = [data[h][0:n] for h in helper] # Get data lists by label
mass_helper   = [mass[h][0:n] for h in helper] # Get kinematics lists by label

target_ = target
data_   = data
mass_   = mass
# Concatenate lists from different labels (in order)
target = ma.concatenate(target_helper)  #NOTE: Important to use ma.concatenate here.
data   = ma.concatenate(data_helper) #NOTE: Important to use ma.concatenate here.
mass   = ma.concatenate(mass_helper) #NOTE: Important to use ma.concatenate here.

#--------------------------------------------------------------------------------------#
# target = target_
# data = data_
# mass = mass_
print("n = ",n)
print("len(target)",len(target))
print("len(data)",len(data))
print("len(mass)",len(mass))
print("target[0] = ",target[0])
print("mass[0] = ",mass[0])
print("data[0][0:15] = ",data[0][0:15])
print("type(target_) = ",type(target_))
print("type(target) = ",type(target))

# Convert data events to DGL Graphs
mydataset = {"data":[], "target":[], "mass":[]}
counter = -1
debugging_counter = 0

for datap in data:

    # Update counter
    counter += 1
    
#     print(np.shape(mc_pa_pid))
#     if 3122 in mc_pa_pid[counter] and 2114 in mc_pa_pid[counter]:
#         debugging_counter += 1
#         print("DEBUGGING: JOINT EVENT")
#         continue #NOTE: CHECK THAT SKIM FOR DELTA IS EXCLUSIVE!

    # Get lists for unique index combinations
    nomask = datap[:,pT_index].count() #IMPORTANT: Just count along single data entry for all particles in event.
    if nomask<2: 
        print("*** WARNING *** Event",counter,"has just 1 particle.  Skipping.")
        print(datap[:,pid_index])
        continue
        
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
#     try:
    bg.ndata['data'] = torch.moveaxis(torch.tensor(ma.array([datap_pT,datap_phi,datap_theta,datap_beta,datap_chi2,datap_pid,datap_status])),(0,1),(1,0))
    if counter < 3: print("bg.ndata['data'] = ",bg.ndata['data'])#DEBUGGING
#         print(ma.array([datap_pT,datap_phi,datap_theta,datap_beta,datap_pid,datap_charge,datap_status,datap_chi2]))
#         break#DEBUGGING
#     except Exception:
#         print("*** WARNING SKIPPING EVENT: ",counter," ***")
#         print(datap_pT.shape,datap_phi.shape,datap_eta.shape,datap_theta.shape)
#         print(datap_chi2.shape,datap_status.shape,datap_charge.shape,datap_beta.shape,datap_pid.shape)
#         continue
    
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
    
    # Add graph and labels to dataset
    mydataset["target"].append(target[counter])
    mydataset["data"].append(bg)
    helper = [target[counter]]
    helper.extend(mass[counter][0]) #NOTE: Just use first entry since lambda multiplicity is not high.
    helper = ma.array(helper,dtype=np.float64)
    mydataset["mass"].append(helper)#NOTE:  Doesn't take into account double match events...
    
        
# Sanity check
print(mydataset["data"][0].ndata)
print(mydataset["data"][0].edata)
print("mydataset[\"mass\"][0] = ",mydataset["mass"][0])

# Shuffle dataset
indices = np.array([i for i in range(len(mydataset["data"]))])
np.random.shuffle(indices) #NOTE: In-place method
mydataset["data"]   = [mydataset["data"][i] for i in indices]
mydataset["target"] = [mydataset["target"][i] for i in indices]
mydataset["mass"]   = [mydataset["mass"][i] for i in indices]

print(counter)
print("DEBUGGING: debugging_counter = ",debugging_counter)#DEBUGGING

# DGL Graph Learning Imports
import dgl #NOTE: for dgl.batch and dgl.unbatch
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_info, load_info

class GraphDataset(DGLDataset):

    """
    Attributes
    ----------

    Methods
    -------

    """

    _url = None
    _sha1_str = None
    mode = "mode"

    def __init__(
        self,
        name="dataset",
        dataset=None,
        inGraphs=None,
        inLabels=None,
        raw_dir=None,
        mode="mode",
        url=None,
        force_reload=False,
        verbose=False,
        num_classes=2
        ):

        """
        Parameters
        ----------
        name : str, optional
            Default : "dataset".
        inGraphs : Tensor(dgl.HeteroGraph), optional
            Default : None.
        inLabels= : Tensor, optional
            Default : None.
        raw_dir : str, optional
            Default : None.
        mode : str, optional
            Default : "mode".
        url : str, optional
            Default : None.
        force_reload : bool, optional
            Default : False.
        verbose : bool, optional
            Default : False.
        num_classes : int, optional
            Default : 2.

        Examples
        --------z

        Notes
        -----
        
        """
        
        self.inGraphs = inGraphs #NOTE: Set these BEFORE calling super.
        self.inLabels = inLabels
        self._url = url
        self.mode = mode
        self.num_classes = num_classes #NOTE: IMPORTANT! You need the self.num_classes variable for the builtin methods of DGLDataset to work!
        super(GraphDataset, self).__init__(name=name,
                                          url=self._url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose
                                          )
        
        
    def process(self):
        mat_path = os.path.join(self.raw_path,self.mode+'_dgl_graph.bin')
        #NOTE: process data to a list of graphs and a list of labels
        
        #DEBUGGING
        ADD = False
        if ADD:
            self.graphs, self.labels = load_graphs(mat_path) #NEED WAY TO ADD
#             self.inGraphs = dgl.unbatch(self.inGraphs)
            print("DEBUGGING: type(self.graphs) = ",type(self.graphs))#DEBUGGING
            print("DEBUGGING: type(self.labels) = ",type(self.labels))#DEBUGGING
            print("type(self.graphs[0]) = ",type(self.graphs[0]))
            print("len(self.graphs) = ",len(self.graphs))
            print("DEBUGGING: type(self.inGraphs) = ",type(self.inGraphs))#DEBUGGING
            print("DEBUGGING: type(self.inLabels) = ",type(self.inLabels))#DEBUGGING
            print("type(self.inGraphs[0]) = ",type(self.inGraphs[0]))
            print("len(self.inGraphs) = ",len(self.inGraphs))
            print("self.labels.keys() = ",self.labels.keys())#DEBUGGING
            print("type(self.labels['labels']) = ",type(self.labels['labels']))
#             print("type(self.inLabels['labels']) = ",type(self.inLabels['labels']))
            self.graphs.extend(self.inGraphs)
            print("NEW LENGTH self.graphs = ",len(self.graphs))#DEBUGGING
            self.labels = torch.cat((self.labels['labels'],self.inLabels),dim=0)
            print("NEW LENGTH self.labels = ",len(self.labels))
#             self.graphs, self.labels = np.concatenate(self.graphs,self.inGraphs), torch.concatenate(self.labels,self.inLabels)
            
            return
        #DEBUGGING
        
        if self.inGraphs is not None and self.inLabels is not None:
            self.graphs, self.labels = self.inGraphs, self.inLabels #DEBUGGING: COMMENTED OUT: torch.LongTensor(self.inLabels)
        else:
            self.graphs, self.labels = load_graphs(mat_path) #NEED WAY TO ADD

    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        save_info(info_path, {'num_classes': self.num_classes})
    
    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        self.num_classes = load_info(info_path)['num_classes']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)
    
    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return self.num_classes


# Name dataset
# dataset_name = 'gangelmc_jobs_Lambda_train_outbending_5_17_22__pT_phi_theta_beta_chi2_pid_status__Normalized'
dataset_name = 'LambdaTrain_jobs_cache_clas12_rga_fall2018_bg50nA_10_7_22__pT_phi_theta_beta_chi2_pid_status__Normalized'
# dataset_name = 'test'

# Load and save data
train_dataset = GraphDataset(dataset_name,inGraphs=mydataset["data"],
                             inLabels=torch.tensor(np.array(mydataset["mass"]),dtype=torch.float64))
train_dataset.process()
train_dataset.save()

print("DONE")

