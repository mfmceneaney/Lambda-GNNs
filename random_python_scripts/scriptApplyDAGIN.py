import numpy as np
import awkward as ak
import hipopy.hipopy as hp
import torch
import dgl
import sys

sys.path.append('.')
from applyGNN import load_model, load_modelDA, createGraph, apply

filename = sys.argv[1]

bank     = "ML::pred"
dtype    = "D" #NOTE: For now all the bank entries have to have the same type.
names    = ["label","pred"]
namesAndTypes = {e:dtype for e in names}
rows = 1 # Chooose a #
# nbatches = 10 # Choose a # #NOTE: NO LONGER NEEDED
step = 1 # Choose a #

file = hp.recreate(filename) #NOTE: ORIGINALLY RECREATE NOT OPEN
file.newTree(bank,namesAndTypes)
file.open() # IMPORTANT!  Open AFTER calling newTree, otherwise the banks will not be written!

nlayers = 3
nmlp = 3
hdim = 193
nmlp_head = 2
hdim_head = 96
dropout = 0.63815
learn_eps = False
npooling = 'mean'
gpooling = 'mean'
device = torch.device('cpu')
path = '/Users/mfm45/drop/tomove/log_optimize_dagnn_delta_fraction_0.2_9_6_22/trial_2022-09-14_LambdaTrain_jobs_cache_clas12_rga_fall2018_bg50nA_9_6_22__pT_phi_theta_beta_chi2_pid_status__Normalized__Delta_fraction_0.2_log_optimize_dagnn_delta_fraction_0.2_9_6_22_139'
name = 'model'

model = load_modelDA(nlayers,nmlp,hdim,nmlp_head,hdim_head,dropout,learn_eps,npooling,gpooling,device,path,name)

counter = 0
with torch.no_grad():
    model.eval()#NOTE: IMPORTANT! OTHERWISE BATCHNORM THROWS ERRORS
    try:
        for m in model.models: m.eval()
    except Exception:
        print("DEBUGGING: THERE WAS AN EXCEPTION")
        pass
    
    for event in file:
        counter += 1

        graph  = createGraph(event)
        result = apply(model,graph)
        label  = result[0][0][0].item()
        maxes  = result[1][0]
        data   = np.array([[label],[maxes[1]]]) #NOTE: IMPORTANT! size=(len(names),rows))

        file.update({bank : data})

    file.close() #IMPORTANT!