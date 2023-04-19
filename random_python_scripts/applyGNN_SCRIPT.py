import numpy as np
import awkward as ak
import hipopy.hipopy as hp
import uproot as ur
import torch
import dgl
import sys
import pandas as pd
import os.path as osp
import argparse

sys.path.append('.')
from applyGNN import load_model, load_modelDA, createGraph, apply

# Create parser
parser = argparse.ArgumentParser(description='Script for applying DGL GIN to HIPO file and outputing results to CSV')
parser.add_argument('--infile', type=str, default='',help='input file name')
parser.add_argument('--outfile',type=str, default='',help='output file name')
parser.add_argument('--tree', type=str, default='t',help='output ROOT file tree name')
parser.add_argument('--step',type=int, default=1000, help='step size for writing events to ROOT')
parser.add_argument('--nlayers',type=int, default=3, help='number of GIN layers')
parser.add_argument('--nmlp',type=int, default=3, help='number of mlp layers')
parser.add_argument('--hdim',type=int, default=64, help='hidden dimension')
parser.add_argument('--dropout',type=float, default=0.5, help='number of GIN layers')
parser.add_argument('--learn_eps',action='store_true',help='learn the epsilon weighting for the center nodes')
parser.add_argument('--npooling',type=str, default='max', help='node pooling type')
parser.add_argument('--gpooling',type=str, default='mean', help='graph pooling type')
parser.add_argument('--path',type=str,default='',help='path to model weights')
parser.add_argument('--name',type=str,default='',help='name of model (currently stored as <name>_weights)')
args = parser.parse_args()

# Set parameters
filename = args.infile #'file.hipo'
outfile  = args.outfile #'test.csv'
treename = args.tree #'t'

# Set model parameters
nlayers = args.nlayers
nmlp = args.nmlp
hdim = args.hdim
dropout = args.dropout
learn_eps = args.learn_eps
npooling = args.npooling
gpooling = args.gpooling
device = torch.device('cpu')
path = osp.expanduser(args.path) #'~/drop/log_TEST_10_17_22'
name = args.name #'model'

# Open HIPO file
file = hp.open(filename)
urfile = ur.recreate(outfile)

# Instantiate model
model = load_model(nlayers,nmlp,hdim,dropout,learn_eps,npooling,gpooling,device,path,name)

# Run model on file
with torch.no_grad():
    model.eval()#NOTE: IMPORTANT! OTHERWISE BATCHNORM THROWS ERRORS
    results = []

    # Loop events
    for idx, event in enumerate(file):

        #NOTE: COULD ADD CHECK HERE FOR PARTICLES IN EVENT AND JUST MATCH TO EVENTS FROM CLAS12ANALYSIS?

        graph = createGraph(event)
        argmax_Y, probs_Y = apply(model,graph) #NOTE: NEED TO ADD OPTION FOR GETTING DEVICE...
        run_number   = event['RUN::config_run'][0][0]
        event_number = event['RUN::config_event'][0][0]
        results.append([run_number,event_number,argmax_Y[0][0],*probs_Y[0]])

        if idx % step == 0:
            # Convert results to dataframe and output to csv file
            results = np.array(results)
            columns = ['run','event','label','result0','result1']
            df = pd.DataFrame(results,columns=columns)
            # df.to_csv(outfile,index=True,index_label='index')
            if treename in urfile.keys(): urfile[treename].extend(df)
            else: urfile[treename] = df
            results = []
    
    # Convert results to dataframe and output to csv file
    results = np.array(results)
    columns = ['run','event','label','result0','result1']
    df = pd.DataFrame(results,columns=columns)
    # df.to_csv(outfile,index=True,index_label='index')
    if treename in urfile.keys(): urfile[treename].extend(df)
    else: urfile[treename] = df

urfile.close()

print("DONE") #DEBUGGING
