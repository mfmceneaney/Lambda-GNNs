###############################
# Matthew McEneaney
# 7/8/21
###############################

from __future__ import absolute_import, division, print_function

# ML Imports
import numpy as np
import numpy.ma as ma
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib as mpl
import matplotlib.pyplot as plt

# DGL Graph Learning Imports
from dgl import save_graphs, load_graphs, batch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_info, load_info

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.data.utils import Subset

# PyTorch Ignite Imports
from ignite.engine import Engine, Events, EventEnum, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import global_step_from_engine, EarlyStopping

# Optuna imports
import optuna
from optuna.samplers import TPESampler

# Fitting imports
import scipy.optimize as opt
from scipy.stats import crystalball

# Utility Imports
import datetime, os

# Local Imports
from models import GIN, HeteroGIN

def get_graph_dataset_info(dataset="",prefix="",batch_size=1024,drop_last=False,shuffle=True,num_workers=0,pin_memory=True, verbose=True):

    # Load training data
    train_dataset = LambdasDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    num_labels = train_dataset.num_labels
    node_feature_dim = train_dataset.graphs[0].ndata["data"].shape[-1]
    train_dataset.load()
    train_dataset = Subset(train_dataset,range(1))

    return num_labels, node_feature_dim

def load_graph_dataset(dataset="",prefix="",split=0.75,max_events=1e5,batch_size=1024,drop_last=False,shuffle=True,num_workers=0,pin_memory=True, verbose=True):

    # Load training data
    train_dataset = LambdasDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    train_dataset.load()
    num_labels = train_dataset.num_labels
    node_feature_dim = train_dataset.graphs[0].ndata["data"].shape[-1]
    index = int(min(len(train_dataset),max_events)*split)
    train_dataset = Subset(train_dataset,range(index))

    # Create training dataloader
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)

    # Load validation data
    val_dataset = LambdasDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    val_dataset.load()
    val_dataset = Subset(train_dataset,range(index,len(val_dataset)))

    # Create testing dataloader
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers)

    return train_loader, val_loader, num_labels, node_feature_dim    

def train(args, model, device, train_loader, val_loader, optimizer, scheduler, criterion, max_epochs,
            dataset="", prefix="", log_interval=10,log_dir="logs/",save_path="logs/model",verbose=True):

    # Make sure log/save directories exist
    try:
        os.mkdir(log_dir+"tb_logs/tmp") #NOTE: Do NOT use os.path.join() here since it requires that the directory exist.
    except Exception:
        if verbose: print("Could not create directory:",os.path.join(log_dir,"tb_logs/tmp"))

    # Show model if requested
    if verbose: print(model)

    # Instatiate torchscript model to save (for importing model to java/c++)
    traced_cell = None

    # Logs for matplotlib plots
    logs={'train':{'loss':[],'accuracy':[],'roc_auc':[]}, 'val':{'loss':[],'accuracy':[],'roc_auc':[]}}

    # Create trainer
    def train_step(engine, batch):
        model.train()
        x, label   = batch
        y = label[:,0].clone().detach().long()
        x      = x.to(device)
        y      = y.to(device)
        y_pred = model(x)
        loss   = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        traced_cell = torch.jit.trace(model, x)
        test_Y = y.clone().detach().float().view(-1, 1) 
        probs_Y = torch.softmax(y_pred, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
        return {
                'y_pred': y_pred,
                'y': y,
                'y_pred_preprocessed': argmax_Y,
                'loss': loss.detach().item(),
                'accuracy': acc
                }

    trainer = Engine(train_step)

    # Add metrics
    accuracy  = Accuracy(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    accuracy.attach(trainer, 'accuracy')
    loss      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
    loss.attach(trainer, 'loss')
    roc_auc   = ROC_AUC(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    roc_auc.attach(trainer,'roc_auc')

    # Create validator
    def val_step(engine, batch):
        model.eval()
        x, label   = batch
        y = label[:,0].clone().detach().long()
        x      = x.to(device)
        y      = y.to(device)
        y_pred = model(x)
        loss   = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        test_Y = y.clone().detach().float().view(-1, 1) 
        probs_Y = torch.softmax(y_pred, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
        model.train()
        return {
                'y_pred': y_pred,
                'y': y,
                'y_pred_preprocessed': argmax_Y,
                'loss': loss.detach().item(),
                'accuracy': acc
                }

    evaluator = Engine(val_step)

    # Add metrics
    accuracy_  = Accuracy(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    accuracy_.attach(evaluator, 'accuracy')
    loss_      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
    loss_.attach(evaluator, 'loss')
    roc_auc_   = ROC_AUC(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    roc_auc_.attach(evaluator,'roc_auc')

    ############################################################################################
    # ADDED EARLY STOPPING
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    handler = EarlyStopping(patience=args.patience, min_delta=args.min_delta, cumulative_delta=args.cumulative_delta, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    evaluator.add_event_handler(Events.COMPLETED, handler)
    ############################################################################################

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(trainer):
        if verbose: print(f"\rEpoch[{trainer.state.epoch}/{max_epochs} : " +
            f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
            f"Loss: {trainer.state.output['loss']:.3f} Accuracy: {trainer.state.output['accuracy']:.3f}",end='')

    @trainer.on(Events.EPOCH_COMPLETED)
    def stepLR(trainer):
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        metrics = evaluator.run(train_loader).metrics
        for metric in metrics.keys(): logs['train'][metric].append(metrics[metric])
        if verbose: print(f"\nTraining Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['loss']:.4f} Avg accuracy: {metrics['accuracy']:.4f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        metrics = evaluator.run(val_loader).metrics
        for metric in metrics.keys(): logs['val'][metric].append(metrics[metric])
        if verbose: print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['loss']:.4f} Avg accuracy: {metrics['accuracy']:.4f}")

    # Create a TensorBoard logger
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training_by_iteration",
        output_transform=lambda x: x["loss"]
    )
        
    # Attach the logger to the evaluator on the training dataset and log Loss, Accuracy metrics after each epoch
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=["loss","accuracy","roc_auc"],
        global_step_transform=global_step_from_engine(trainer),
    )

    # Attach the logger to the evaluator on the validation dataset and log Loss, Accuracy metrics after
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["loss","accuracy","roc_auc"],
        global_step_transform=global_step_from_engine(evaluator)
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        param_name='lr'  # optional
    )

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)
    tb_logger.close() #IMPORTANT!
    if save_path!="":
        torch.save(model.to('cpu').state_dict(), os.path.join(log_dir,save_path))
        traced_cell.save(os.path.join(log_dir,save_path+'.zip'))

    # Create training/validation loss plot
    f = plt.figure()
    plt.subplot()
    plt.title('Loss per epoch')
    plt.plot(logs['train']['loss'],label="training")
    plt.plot(logs['val']['loss'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    f.savefig(os.path.join(log_dir,'training_metrics_loss_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))

    # Create training/validation accuracy plot
    f = plt.figure()
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['accuracy'],label="training")
    plt.plot(logs['val']['accuracy'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    f.savefig(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))
    
def evaluate(model,device,dataset="", prefix="", split=0.75, max_events=1e10, log_dir="logs/",verbose=True):

    # Load validation data
    test_dataset = LambdasDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    test_dataset.load()
    test_dataset = Subset(test_dataset,range(int(min(len(test_dataset),max_events)*split)))

    model.eval()
    model      = model.to(device)
    test_bg    = batch(test_dataset.dataset.graphs)
    test_Y     = test_dataset.labels[:,0].clone().detach().float().view(-1, 1) #IMPORTANT: keep .view() here
    test_bg    = test_bg.to(device)
    test_Y     = test_Y.to(device)
    prediction = model(test_bg)
    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)
    if verbose: print('Accuracy of predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

    # Copy arrays back to CPU
    test_Y   = test_Y.cpu()
    probs_Y  = probs_Y.cpu()
    argmax_Y = argmax_Y.cpu()

    # Get separated mass distributions
    mass_sig_Y    = ma.array(test_dataset.labels[:,1].clone().detach().float(),mask=~(argmax_Y == 1))
    mass_bg_Y     = ma.array(test_dataset.labels[:,1].clone().detach().float(),mask=~(argmax_Y == 0))

    # Get false-positive true-negatives and vice versa
    mass_sig_true  = ma.array(test_dataset.labels[:,1].clone().detach().float(),
                                mask=np.logical_or(~(torch.squeeze(argmax_Y) == 1),~(torch.squeeze(argmax_Y) == test_dataset.labels[:,0].clone().detach().float())))
    mass_bg_true   = ma.array(test_dataset.labels[:,1].clone().detach().float(),
                                mask=np.logical_or(~(torch.squeeze(argmax_Y) == 0),~(torch.squeeze(argmax_Y) == test_dataset.labels[:,0].clone().detach().float())))
    mass_sig_false = ma.array(test_dataset.labels[:,1].clone().detach().float(),
                                mask=np.logical_or(~(torch.squeeze(argmax_Y) == 1),~(torch.squeeze(argmax_Y) != test_dataset.labels[:,0].clone().detach().float())))
    mass_bg_false  = ma.array(test_dataset.labels[:,1].clone().detach().float(),
                                mask=np.logical_or(~(torch.squeeze(argmax_Y) == 0),~(torch.squeeze(argmax_Y) != test_dataset.labels[:,0].clone().detach().float())))

    # Get separated mass distributions MC-Matched
    mass_sig_MC   = ma.array(test_dataset.labels[:,1].clone().detach().float(),mask=~(test_dataset.labels[:,0] == 1))
    mass_bg_MC    = ma.array(test_dataset.labels[:,1].clone().detach().float(),mask=~(test_dataset.labels[:,0] == 0))

    ##################################################
    # Define fit function
    def func(x, N, beta, m, loc, scale, A, B, C):
        return N*crystalball.pdf(-x, beta, m, -loc, scale) + A*(1 - B*(x - C)**2)
        
    def sig(x, N, beta, m, loc, scale):
        return N*crystalball.pdf(-x, beta, m, -loc, scale)
        
    def bg(x, A, B, C):
        return A*(1 - B*(x - C)**2)
    ##################################################

    # Plot mass decisions separated into signal/background
    bins = 100
    low_high = (1.08,1.24)#(1.1,1.13)
    f = plt.figure(figsize=(16,10))
    plt.title('Separated mass distribution')
    hdata = plt.hist(mass_sig_Y[~mass_sig_Y.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    
    # Fit output of NN
    N, beta, m, loc, scale, A, B, C = 500, 1, 1.112, 1.115, 0.008, hdata[0][-1], 37, 1.24
    d_N, d_beta, d_m, d_loc, d_scale, d_A, d_B, d_C = N/0.01, beta/0.1, m/0.1, loc/0.1, scale/0.01, A/10, B/0.1, C/0.1
    parsMin = [N-d_N, beta-d_beta, m-d_m, loc-d_loc, scale-d_scale, A-d_A, B-d_B, C-d_C]
    parsMax = [N+d_N, beta+d_beta, m+d_m, loc+d_loc, scale+d_scale, A+d_A, B+d_B, C+d_C]
    optParams, pcov = opt.curve_fit(func, hdata[1][:-1], hdata[0], method='trf', bounds=(parsMin,parsMax))

    # Plot fit
    x = np.linspace(low_high[0],low_high[1],bins)#mass_sig_Y[~mass_sig_Y.mask]
    y = hdata[0]
    plt.plot(x, func(x, *optParams), color='r')
    plt.plot(x, sig(x, *optParams[0:5]), color='tab:purple')
    plt.plot(x, bg(x, *optParams[5:]), color='b')
    plt.hist(x, weights=y-bg(x, *optParams[5:]), bins=bins, range=low_high, histtype='step', alpha=0.5, color='b')

    # Setup legend entries for fit info
    lg = "Fit Info\n-------------------------\n"
    lg += f"N = {round(optParams[0],0)}±{round(pcov[0,0],5)}\n"
    lg += f"α = {round(optParams[1],3)}±{round(pcov[1,1],5)}\n"
    lg += f"n = {round(optParams[2],3)}±{round(pcov[2,2],5)}\n"
    lg += f"μ = {round(optParams[3],5)}±{round(pcov[3,3],5)}\n"
    lg += f"σ = {round(optParams[4],5)}±{round(pcov[4,4],5)}\n"
    lg += f"A = {round(optParams[5],0)}±{round(pcov[5,6],5)}\n"
    lg += f"β = {round(optParams[6],0)}±{round(pcov[6,6],5)}\n"
    lg += f"M = {round(optParams[7],2)}±{round(pcov[7,7],5)}\n"
    plt.text(low_high[1]-(low_high[1]-low_high[0])/3,2/3*max(hdata[0]),lg,fontsize=16,linespacing=1.5)

    # Show the graph
    plt.hist(mass_bg_Y[~mass_bg_Y.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'eval_metrics_mass_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot mass decisions separated into signal/background
    bins = 100
    # low_high = (1.1,1.13)
    f = plt.figure()
    plt.title('Separated mass distribution')
    plt.hist(mass_sig_Y[~mass_sig_Y.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    plt.hist(mass_bg_Y[~mass_bg_Y.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'test_metrics_mass_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot correct mass decisions separated into signal/background
    bins = 100
    # low_high = (1.1,1.13)
    f = plt.figure()
    plt.title('Separated mass distribution (true)')
    plt.hist(mass_sig_true[~mass_sig_true.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    plt.hist(mass_bg_true[~mass_bg_true.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'test_metrics_mass_true_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot incorrect mass decisions separated into signal/background
    bins = 100
    # low_high = (1.1,1.13)
    f = plt.figure()
    plt.title('Separated mass distribution (false)')
    plt.hist(mass_sig_false[~mass_sig_false.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    plt.hist(mass_bg_false[~mass_bg_false.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'test_metrics_mass_false_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot MC-Matched distributions
    bins = 100
    # low_high = (1.1,1.13)
    f = plt.figure()
    plt.title('Separated mass distribution MC-matched')
    plt.hist(mass_sig_MC[~mass_sig_MC.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    plt.hist(mass_bg_MC[~mass_bg_MC.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'mc_matched_mass_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot MC-Matched distributions for NN-identified signal
    bins = 100
    # low_high = (1.1,1.13)
    f = plt.figure()
    plt.title('NN-identified signal mass distribution MC-matched')
    plt.hist(mass_sig_true[~mass_sig_true.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='true')
    plt.hist(mass_sig_false[~mass_sig_false.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='false')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'mc_matched_nn_sig_mass_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot MC-Matched distributions for NN-identified background
    bins = 100
    # low_high = (1.1,1.13)
    f = plt.figure()
    plt.title('NN-identified bg mass distribution MC-matched')
    plt.hist(mass_bg_true[~mass_bg_true.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='true')
    plt.hist(mass_bg_false[~mass_bg_false.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='false')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'mc_matched_nn_bg_mass_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Get ROC curve
    pfn_fp, pfn_tp, threshs = roc_curve(test_Y.detach().numpy(), probs_Y[:,1].detach().numpy())

    # Get area under the ROC curve
    auc = roc_auc_score(test_Y.detach().numpy(), probs_Y[:,1].detach().numpy())
    if verbose: print(f'AUC = {auc:.4f}')

    # Create matplotlib plots for ROC curve and testing decisions
    f = plt.figure()

    # Get some nicer plot settings 
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True

    # Plot the ROC curve
    plt.plot(pfn_tp, 1-pfn_fp, '-', color='black', label=model.name)

    # axes labels
    plt.xlabel('Lambda Event Efficiency')
    plt.ylabel('Background Rejection')

    # axes limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # make legend and show plot
    plt.legend([model.name+f": AUC={auc:.4f}"],loc='lower left', frameon=False)
    f.savefig(os.path.join(log_dir,model.name+"_ROC_"+datetime.datetime.now().strftime("%F")+dataset+".png"))

    ##########################################################
    # Plot testing decisions
    bins = 100
    low = min(np.min(p) for p in probs_Y[:,1].detach().numpy())
    high = max(np.max(p) for p in probs_Y[:,0].detach().numpy())
    low_high = (low,high)
    f = plt.figure()
    plt.clf()
    plt.hist(probs_Y[:,1].detach().numpy(), color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist1')
    plt.hist(probs_Y[:,0].detach().numpy(), color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist2')
    plt.xlabel('output')
    plt.ylabel('counts')
    f.savefig(os.path.join(log_dir,model.name+"_test_decisions_"+datetime.datetime.now().strftime("%F")+dataset+".png"))

def optimization_study(args,log_interval=10,log_dir="logs/",save_path="torch_models",verbose=True):
    #NOTE: As of right now log_dir='logs/' should end with the slash

    # Load validation data
    test_dataset = LambdasDataset(args.prefix+args.dataset)
    test_dataset.load()
    test_dataset = Subset(test_dataset,range(int(len(test_dataset)*args.split)))

    def objective(trial):

        # Get parameter suggestions for trial
        batch_size = args.batch[0] if args.batch[0] == args.batch[1] else trial.suggest_int("batch_size",args.batch[0],args.batch[1]) 
        nlayers = args.nlayers[0] if args.nlayers[0] == args.nlayers[1] else trial.suggest_int("nlayers",args.nlayers[0],args.nlayers[1])
        nmlp  = args.nmlp[0] if args.nmlp[0] == args.nmlp[1] else trial.suggest_int("nmlp",args.nmlp[0],args.nmlp[1])
        hdim  = args.hdim[0] if args.hdim[0] == args.hdim[1] else trial.suggest_int("hdim",args.hdim[0],args.hdim[1])
        do    = args.dropout[0] if args.dropout[0] == args.dropout[1] else trial.suggest_float("do",args.dropout[0],args.dropout[1])
        lr    = args.lr[0] if args.lr[0] == args.lr[1] else trial.suggest_float("lr",args.lr[0],args.lr[1],log=True)
        step  = args.step[0] if args.step[0] == args.step[1] else trial.suggest_int("step",args.step[0],args.step[1])
        gamma = args.gamma[0] if args.gamma[0] == args.gamma[1] else trial.suggest_float("gamma",args.gamma[0],args.gamma[1])
        max_epochs = args.epochs

        # Setup data and model
        train_loader, val_loader, nclasses, nfeatures = load_graph_dataset(dataset=args.dataset, split=args.split, max_events = args.max_events,
                                                        num_workers=args.nworkers, batch_size=batch_size)

        # Initiate model and optimizer, scheduler, loss
        model = GIN(nlayers,nmlp,nfeatures,hdim,nclasses,do,args.learn_eps,args.npooling,args.gpooling).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
        criterion = nn.CrossEntropyLoss()

        # Make sure log/save directories exist
        trialdir = 'trial_'+datetime.datetime.now().strftime("%F")+'_'+args.dataset+'_'+args.study_name+'_'+str(trial.number)
        try:
            os.mkdir(args.log+'/'+trialdir) #NOTE: Do NOT use os.path.join() here since it requires that the directory already exist.
        except FileExistsError:
            if args.verbose: print("Directory:",os.path.join(args.log,trialdir))
        trialdir = os.path.join(args.log,trialdir)

        # Show model if requested
        if args.verbose: print(model)

        # Instatiate torchscript model to save (for importing model to java/c++)
        traced_cell = None

        # Logs for matplotlib plots
        logs={'train':{'loss':[],'accuracy':[],'roc_auc':[]}, 'val':{'loss':[],'accuracy':[],'roc_auc':[]}}

        # Create trainer
        def train_step(engine, batch):
            model.train()
            x, label   = batch
            y = label[:,0].clone().detach().long()
            x      = x.to(args.device)
            y      = y.to(args.device)
            y_pred = model(x)
            loss   = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            traced_cell = torch.jit.trace(model, x)
            test_Y = y.clone().detach().float().view(-1, 1) 
            probs_Y = torch.softmax(y_pred, 1)
            argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
            acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
            return {
                    'y_pred': y_pred,
                    'y': y,
                    'y_pred_preprocessed': argmax_Y,
                    'loss': loss.detach().item(),
                    'accuracy': acc
                    }

        trainer = Engine(train_step)

        # Add metrics
        accuracy  = Accuracy(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
        accuracy.attach(trainer, 'accuracy')
        loss      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
        loss.attach(trainer, 'loss')
        roc_auc   = ROC_AUC(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
        roc_auc.attach(trainer,'roc_auc')

        # Create validator
        def val_step(engine, batch):
            model.eval()
            x, label   = batch
            y = label[:,0].clone().detach().long()
            x      = x.to(args.device)
            y      = y.to(args.device)
            y_pred = model(x)
            loss   = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_Y = y.clone().detach().float().view(-1, 1) 
            probs_Y = torch.softmax(y_pred, 1)
            argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
            acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
            model.train()
            return {
                    'y_pred': y_pred,
                    'y': y,
                    'y_pred_preprocessed': argmax_Y,
                    'loss': loss.detach().item(),
                    'accuracy': acc
                    }

        evaluator = Engine(val_step)

        # Add metrics
        accuracy_  = Accuracy(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
        accuracy_.attach(evaluator, 'accuracy')
        loss_      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
        loss_.attach(evaluator, 'loss')
        roc_auc_   = ROC_AUC(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
        roc_auc_.attach(evaluator,'roc_auc')

        # Register a pruning handler to the evaluator.
        pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(trial, "accuracy", evaluator) #ORIGINALLY TRAINER
        evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

        # Add early stopping
        def score_function(engine):
            val_loss = engine.state.metrics['loss']
            return -val_loss

        handler = EarlyStopping(patience=args.patience, min_delta=args.min_delta, cumulative_delta=args.cumulative_delta, score_function=score_function, trainer=trainer)
        # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
        evaluator.add_event_handler(Events.COMPLETED, handler)

        @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
        def log_training_loss(trainer):
            if verbose: print(f"\rEpoch[{trainer.state.epoch}/{max_epochs} : " +
                f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
                f"Loss: {trainer.state.output['loss']:.3f} Accuracy: {trainer.state.output['accuracy']:.3f}",end='')

        @trainer.on(Events.EPOCH_COMPLETED)
        def stepLR(trainer):
            scheduler.step()

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            metrics = evaluator.run(train_loader).metrics
            for metric in metrics.keys(): logs['train'][metric].append(metrics[metric])
            if verbose: print(f"\nTraining Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['loss']:.4f} Avg accuracy: {metrics['accuracy']:.4f}")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            metrics = evaluator.run(val_loader).metrics
            for metric in metrics.keys(): logs['val'][metric].append(metrics[metric])
            if verbose: print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['loss']:.4f} Avg accuracy: {metrics['accuracy']:.4f}")

        # Create a TensorBoard logger
        try: os.mkdir(trialdir+"/tb_logs")
        except FileExistsError:
            print('TB directory:',trialdir+"/tb_logs","already exists!")
        tb_logger = TensorboardLogger(log_dir=trialdir+"/tb_logs")

        # Attach the logger to the trainer to log training loss at each iteration
        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training_by_iteration",
            output_transform=lambda x: x["loss"]
        )
            
        # Attach the logger to the evaluator on the training dataset and log Loss, Accuracy metrics after each epoch
        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            tag="training",
            metric_names=["loss","accuracy","roc_auc"],
            global_step_transform=global_step_from_engine(trainer),
        )

        # Attach the logger to the evaluator on the validation dataset and log Loss, Accuracy metrics after
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names=["loss","accuracy","roc_auc"],
            global_step_transform=global_step_from_engine(evaluator)
        )

        # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
        tb_logger.attach_opt_params_handler(
            trainer,
            event_name=Events.ITERATION_STARTED,
            optimizer=optimizer,
            param_name='lr'  # optional
        )

        # Run training loop
        trainer.run(train_loader, max_epochs=max_epochs)
        tb_logger.close() #IMPORTANT!
        if save_path!="":
            torch.save(model.to('cpu').state_dict(),os.path.join(trialdir,save_path+args.study_name+'_'+str(trial.number)))
            traced_cell.save(os.path.join(trialdir,save_path+args.study_name+'_'+str(trial.number)+'.zip'))

        # Create training/validation loss plot
        f = plt.figure()
        plt.subplot()
        plt.title('Loss per epoch')
        plt.plot(logs['train']['loss'],label="training")
        plt.plot(logs['val']['loss'],label="validation")
        plt.legend(loc='best', frameon=False)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        f.savefig(os.path.join(trialdir,'training_metrics_loss_'+datetime.datetime.now().strftime("%F")+"_"+args.dataset+"_nEps"+str(max_epochs)+'.png'))

        # Create training/validation accuracy plot
        f = plt.figure()
        plt.subplot()
        plt.title('Accuracy per epoch')
        plt.plot(logs['train']['accuracy'],label="training")
        plt.plot(logs['val']['accuracy'],label="validation")
        plt.legend(loc='best', frameon=False)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        f.savefig(os.path.join(trialdir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+"_"+args.dataset+"_nEps"+str(max_epochs)+'.png'))

        # Evaluate model
        model.eval()
        test_bg    = batch(test_dataset.dataset.graphs)
        test_Y     = test_dataset.labels[:,0].clone().detach().float().view(-1, 1) #IMPORTANT: keep .view() here
        test_bg    = test_bg.to(args.device)
        test_Y     = test_Y.to(args.device)
        prediction = model(test_bg)
        probs_Y    = torch.softmax(prediction, 1)
        argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)
        test_acc   = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
        if verbose: print('Accuracy of predictions on the test set: {:4f}%'.format(
            (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

        # Copy arrays back to CPU
        test_Y   = test_Y.cpu()
        probs_Y  = probs_Y.cpu()
        argmax_Y = argmax_Y.cpu()

        # Get separated mass distributions
        mass_sig_Y    = ma.array(test_dataset.labels[:,1].clone().detach().float(),mask=~(argmax_Y == 1))
        mass_bg_Y     = ma.array(test_dataset.labels[:,1].clone().detach().float(),mask=~(argmax_Y == 0))

        # Get false-positive true-negatives and vice versa
        mass_sig_true  = ma.array(test_dataset.labels[:,1].clone().detach().float(),
                                    mask=np.logical_or(~(torch.squeeze(argmax_Y) == 1),~(torch.squeeze(argmax_Y) == test_dataset.labels[:,0].clone().detach().float())))
        mass_bg_true   = ma.array(test_dataset.labels[:,1].clone().detach().float(),
                                    mask=np.logical_or(~(torch.squeeze(argmax_Y) == 0),~(torch.squeeze(argmax_Y) == test_dataset.labels[:,0].clone().detach().float())))
        mass_sig_false = ma.array(test_dataset.labels[:,1].clone().detach().float(),
                                    mask=np.logical_or(~(torch.squeeze(argmax_Y) == 1),~(torch.squeeze(argmax_Y) != test_dataset.labels[:,0].clone().detach().float())))
        mass_bg_false  = ma.array(test_dataset.labels[:,1].clone().detach().float(),
                                    mask=np.logical_or(~(torch.squeeze(argmax_Y) == 0),~(torch.squeeze(argmax_Y) != test_dataset.labels[:,0].clone().detach().float())))

        # Get separated mass distributions MC-Matched
        mass_sig_MC   = ma.array(test_dataset.labels[:,1].clone().detach().float(),mask=~(test_dataset.labels[:,0] == 1))
        mass_bg_MC    = ma.array(test_dataset.labels[:,1].clone().detach().float(),mask=~(test_dataset.labels[:,0] == 0))

        # Plot mass decisions separated into signal/background
        bins = 100
        low_high = (1.1,1.13)
        f = plt.figure()
        plt.title('Separated mass distribution')
        plt.hist(mass_sig_Y[~mass_sig_Y.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
        plt.hist(mass_bg_Y[~mass_bg_Y.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
        plt.legend(loc='upper left', frameon=False)
        plt.ylabel('Counts')
        plt.xlabel('Invariant mass (GeV)')
        f.savefig(os.path.join(trialdir,'test_metrics_mass_'+datetime.datetime.now().strftime("%F")+args.dataset+'.png'))

        # Plot correct mass decisions separated into signal/background
        bins = 100
        # low_high = (1.1,1.13)
        f = plt.figure()
        plt.title('Separated mass distribution (true)')
        plt.hist(mass_sig_true[~mass_sig_true.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
        plt.hist(mass_bg_true[~mass_bg_true.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
        plt.legend(loc='upper left', frameon=False)
        plt.ylabel('Counts')
        plt.xlabel('Invariant mass (GeV)')
        f.savefig(os.path.join(trialdir,'test_metrics_mass_true_'+datetime.datetime.now().strftime("%F")+args.dataset+'.png'))

        # Plot incorrect mass decisions separated into signal/background
        bins = 100
        # low_high = (1.1,1.13)
        f = plt.figure()
        plt.title('Separated mass distribution (false)')
        plt.hist(mass_sig_false[~mass_sig_false.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
        plt.hist(mass_bg_false[~mass_bg_false.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
        plt.legend(loc='upper left', frameon=False)
        plt.ylabel('Counts')
        plt.xlabel('Invariant mass (GeV)')
        f.savefig(os.path.join(trialdir,'test_metrics_mass_false_'+datetime.datetime.now().strftime("%F")+args.dataset+'.png'))

        # Plot MC-Matched distributions
        bins = 100
        # low_high = (1.1,1.13)
        f = plt.figure()
        plt.title('Separated mass distribution MC-matched')
        plt.hist(mass_sig_MC[~mass_sig_MC.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
        plt.hist(mass_bg_MC[~mass_bg_MC.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
        plt.legend(loc='upper left', frameon=False)
        plt.ylabel('Counts')
        plt.xlabel('Invariant mass (GeV)')
        f.savefig(os.path.join(trialdir,'mc_matched_mass_'+datetime.datetime.now().strftime("%F")+args.dataset+'.png'))

        # Plot MC-Matched distributions for NN-identified signal
        bins = 100
        # low_high = (1.1,1.13)
        f = plt.figure()
        plt.title('NN-identified signal mass distribution MC-matched')
        plt.hist(mass_sig_true[~mass_sig_true.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='true')
        plt.hist(mass_sig_false[~mass_sig_false.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='false')
        plt.legend(loc='upper left', frameon=False)
        plt.ylabel('Counts')
        plt.xlabel('Invariant mass (GeV)')
        f.savefig(os.path.join(trialdir,'mc_matched_nn_sig_mass_'+datetime.datetime.now().strftime("%F")+args.dataset+'.png'))

        # Plot MC-Matched distributions for NN-identified background
        bins = 100
        # low_high = (1.1,1.13)
        f = plt.figure()
        plt.title('NN-identified bg mass distribution MC-matched')
        plt.hist(mass_bg_true[~mass_bg_true.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='true')
        plt.hist(mass_bg_false[~mass_bg_false.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='false')
        plt.legend(loc='upper left', frameon=False)
        plt.ylabel('Counts')
        plt.xlabel('Invariant mass (GeV)')
        f.savefig(os.path.join(trialdir,'mc_matched_nn_bg_mass_'+datetime.datetime.now().strftime("%F")+args.dataset+'.png'))

        # Get ROC curve
        pfn_fp, pfn_tp, threshs = roc_curve(test_Y.detach().numpy(), probs_Y[:,1].detach().numpy())

        # Get area under the ROC curve
        auc = roc_auc_score(test_Y.detach().numpy(), probs_Y[:,1].detach().numpy())
        if verbose: print(f'AUC = {auc:.4f}')

        # Create matplotlib plots for ROC curve and testing decisions
        f = plt.figure()

        # Get some nicer plot settings 
        plt.rcParams['figure.figsize'] = (4,4)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['figure.autolayout'] = True

        # Plot the ROC curve
        plt.plot(pfn_tp, 1-pfn_fp, '-', color='black', label=model.name)

        # axes labels
        plt.xlabel('Lambda Event Efficiency')
        plt.ylabel('Background Rejection')

        # axes limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # make legend and show plot
        plt.legend([model.name+f": AUC={auc:.4f}"],loc='lower left', frameon=False)
        f.savefig(os.path.join(trialdir,model.name+"_ROC_"+datetime.datetime.now().strftime("%F")+args.dataset+".png"))

        # Plot testing decisions
        bins = 100
        low = min(np.min(p) for p in probs_Y[:,1].detach().numpy())
        high = max(np.max(p) for p in probs_Y[:,0].detach().numpy())
        low_high = (low,high)
        f = plt.figure()
        plt.clf()
        plt.hist(probs_Y[:,1].detach().numpy(), color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist1')
        plt.hist(probs_Y[:,0].detach().numpy(), color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist2')
        plt.xlabel('output')
        plt.ylabel('counts')
        f.savefig(os.path.join(trialdir,model.name+"_test_decisions_"+datetime.datetime.now().strftime("%F")+args.dataset+".png"))

        # Close figures: #NOTE: Important for memory!
        plt.close('all')

        return test_acc

    ##### MAIN PART #####
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    sampler = TPESampler() #TODO: Add command line option for selecting different sampler types.
    study = optuna.create_study(sampler=sampler,direction="maximize", pruner=pruner)
    if args.db_path is not None:
        study = optuna.load_study(study_name=args.study_name, storage='sqlite:///'+args.db_path) #TODO: Add options for different SQL programs: Postgre, MySQL, etc.
    study.optimize(objective, n_trials=args.ntrials, timeout=args.timeout)
    trial = study.best_trial

    if verbose:
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

def evaluate_on_data(model,device,dataset="", prefix="", split=0.1, log_dir="logs/",verbose=True):

    # Load validation data
    test_dataset = LambdasDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    test_dataset.load()
    test_dataset = Subset(test_dataset,range(int(len(test_dataset)*split)))

    model.eval()
    model      = model.to(device)
    test_bg    = batch(test_dataset.dataset.graphs)
    test_bg    = test_bg.to(device)
    prediction = model(test_bg)
    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)

    # Copy arrays back to CPU
    probs_Y  = probs_Y.cpu()
    argmax_Y = argmax_Y.cpu()

    # Get separated mass distributions
    mass_sig_Y    = ma.array(test_dataset.labels[:,0].clone().detach().float(),mask=~(argmax_Y == 1))
    mass_bg_Y     = ma.array(test_dataset.labels[:,0].clone().detach().float(),mask=~(argmax_Y == 0))

    ##################################################
    # Define fit function
    def func(x, N, beta, m, loc, scale, A, B, C):
        return N*crystalball.pdf(-x, beta, m, -loc, scale) + A*(1 - B*(x - C)**2)
        
    def sig(x, N, beta, m, loc, scale):
        return N*crystalball.pdf(-x, beta, m, -loc, scale)
        
    def bg(x, A, B, C):
        return A*(1 - B*(x - C)**2)
    ##################################################

    # Plot mass decisions separated into signal/background
    bins = 100
    low_high = (1.08,1.24)#(1.1,1.13)
    f = plt.figure(figsize=(16,10))
    plt.title('Separated mass distribution')
    hdata = plt.hist(mass_sig_Y[~mass_sig_Y.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    
    # Fit output of NN
    N, beta, m, loc, scale, A, B, C = 500, 1, 1.112, 1.115, 0.008, hdata[0][-1], 37, 1.24
    d_N, d_beta, d_m, d_loc, d_scale, d_A, d_B, d_C = N/0.01, beta/0.1, m/0.1, loc/0.1, scale/0.01, A/10, B/0.1, C/0.1
    parsMin = [N-d_N, beta-d_beta, m-d_m, loc-d_loc, scale-d_scale, A-d_A, B-d_B, C-d_C]
    parsMax = [N+d_N, beta+d_beta, m+d_m, loc+d_loc, scale+d_scale, A+d_A, B+d_B, C+d_C]
    optParams, pcov = opt.curve_fit(func, hdata[1][:-1], hdata[0], method='trf', bounds=(parsMin,parsMax))

    # Plot fit
    x = x = np.linspace(low_high[0],low_high[1],bins)#mass_sig_Y[~mass_sig_Y.mask]
    y = hdata[0]
    plt.plot(x, func(x, *optParams), color='r')
    plt.plot(x, sig(x, *optParams[0:5]), color='tab:purple')
    plt.plot(x, bg(x, *optParams[5:]), color='b')
    plt.hist(x, weights=y-bg(x, *optParams[5:]), bins=bins, range=low_high, histtype='step', alpha=0.5, color='b')

    # Setup legend entries for fit info
    lg = "Fit Info\n-------------------------\n"
    lg += f"N = {round(optParams[0],0)}±{round(pcov[0,0],5)}\n"
    lg += f"α = {round(optParams[1],3)}±{round(pcov[1,1],5)}\n"
    lg += f"n = {round(optParams[2],3)}±{round(pcov[2,2],5)}\n"
    lg += f"μ = {round(optParams[3],5)}±{round(pcov[3,3],5)}\n"
    lg += f"σ = {round(optParams[4],5)}±{round(pcov[4,4],5)}\n"
    lg += f"A = {round(optParams[5],0)}±{round(pcov[5,6],5)}\n"
    lg += f"β = {round(optParams[6],0)}±{round(pcov[6,6],5)}\n"
    lg += f"M = {round(optParams[7],2)}±{round(pcov[7,7],5)}\n"
    plt.text(low_high[1]-(low_high[1]-low_high[0])/3,2/3*max(hdata[0]),lg,fontsize=16,linespacing=1.5)

    # Show the graph
    plt.hist(mass_bg_Y[~mass_bg_Y.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'eval_metrics_mass_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    ##########################################################
    # Plot testing decisions
    bins = 100
    low = min(np.min(p) for p in probs_Y[:,1].detach().numpy())
    high = max(np.max(p) for p in probs_Y[:,0].detach().numpy())
    low_high = (low,high)
    f = plt.figure()
    plt.clf()
    plt.hist(probs_Y[:,1].detach().numpy(), color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist1')
    plt.hist(probs_Y[:,0].detach().numpy(), color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist2')
    plt.xlabel('output')
    plt.ylabel('counts')
    f.savefig(os.path.join(log_dir,model.name+"_eval_decisions_"+datetime.datetime.now().strftime("%F")+dataset+".png"))

# Define dataset class
class LambdasDataset(DGLDataset):
    _url = None
    _sha1_str = None
    mode = "mode"
    num_classes = 2
    dataset = None

    def __init__(self, name, dataset=None, raw_dir=None, force_reload=False, verbose=False):
        self.dataset = dataset
        super(LambdasDataset, self).__init__(name=name,
                                          url=self._url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def process(self):
        mat_path = os.path.join(self.raw_path,self.mode+'_dgl_graph.bin')
        # process data to a list of graphs and a list of labels
        if self.dataset != None:
            self.graphs, self.labels = self.dataset["data"], torch.LongTensor(self.dataset["target"])
        else:
            self.graphs, self.labels = load_graphs(mat_path)

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
        return 2
