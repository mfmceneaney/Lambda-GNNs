#--------------------------------------------------#
# Description: Utility functions and classes for 
#   training DGL GNNs.
# Author: Matthew McEneaney
#--------------------------------------------------#

# ML Imports
import numpy as np
import numpy.ma as ma
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib as mpl
import matplotlib.pyplot as plt

# DGL Graph Learning Imports
import dgl #NOTE: for dgl.batch and dgl.unbatch
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_info, load_info, Subset

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# PyTorch Ignite Imports
from ignite.engine import Engine, Events, EventEnum, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import global_step_from_engine, EarlyStopping

# Optuna Imports
import optuna
from optuna.samplers import TPESampler

# Fitting Imports
import scipy.optimize as opt
from scipy.stats import crystalball

# MoDe Loss Imports
from modeloss.pytorch import MoDeLoss

# Utility Imports
import datetime, os, itertools, json

# Local Imports
from models import GIN, HeteroGIN, Classifier, Discriminator, MLP, Concatenate, MLP_SIGMOID

#------------------------- Functions -------------------------#

def get_graph_dataset_info(
    dataset="",
    prefix="",
    key="data",
    ekey=""
    ):

    """
    Parameters
    ----------
    dataset : str, optional
        Default : "".
    prefix : str, optional
        Default : "".
    key : str, optional
        Default : "data".
    ekey : str, optional
        Default : "".

    Returns
    -------
    num_labels : int
        Number of classification labels for dataset
    node_feature_dim : int
        Length of tensors in graph node data
    edge_feature_dim : int
        Length of tensors in graph edge data

    Examples
    --------

    Notes
    -----

    """

    # Load training data
    train_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    num_labels = train_dataset.num_labels
    node_feature_dim = train_dataset.graphs[0].ndata[key].shape[-1]  if  key != '' else 0
    edge_feature_dim = train_dataset.graphs[0].edata[ekey].shape[-1] if ekey != '' else 0
    train_dataset.load()
    train_dataset = Subset(train_dataset,range(1))

    return num_labels, node_feature_dim, edge_feature_dim

def load_graph_dataset(
    dataset="",
    prefix="",
    key="data",
    ekey="",
    split=0.75,
    max_events=1e5,
    indices=None,
    batch_size=1024,
    drop_last=False,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    verbose=True
    ):

    """
    Parameters
    ----------
    dataset : string, optional
        Default : 1024.
    prefix : string, optional
        Default : False.
    indices : tuple, optional
        Tuple of start and stop indices to use
    key : string, optional
        Default : "data".
    ekey : string, optional
        Default : "".
    split : float, optional
        Default : 0.75.
    max_events : int, optional
        Default : 1e5.
    indices : tuple, optional
        Default : None.
    batch_size : int, optional
        Default : 1024.
    drop_last : bool, optional
        Default : False.
    shuffle : bool, optional
        Default : False.
    num_workers : int, optional
        Default : 0.
    pin_memory : bool, optional
        Default : True.
    verbose : bool, optional
        Default : True

    Returns
    -------
    train_loader : dgl.GraphDataLoader
        Dataloader for training data
    val_loader : dgl.GraphDataLoader
        Dataloader for validation data
    eval_loader : dgl.GraphDataLoader
        Dataloader for evaluation data, only returned if >3 indices specified
    num_labels : int
        Number of classification labels for dataset
    node_feature_dim : int
        Length of tensors in graph node data
    edge_feature_dim : int
        Length of tensors in graph edge data

    Examples
    --------

    Notes
    -----
    Load a graph dataset into training and validation loaders based on split fraction.
    """

    # Load training data
    this_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    this_dataset.load()
    num_labels = this_dataset.num_labels
    node_feature_dim = this_dataset.graphs[0].ndata[key].shape[-1]  if  key != '' else 0
    edge_feature_dim = this_dataset.graphs[0].edata[ekey].shape[-1] if ekey != '' else 0

    # Shuffle entire dataset
    if shuffle: this_dataset.shuffle() #TODO: Make the shuffling to dataloading non-biased???

    # Get training subset
    if indices is not None:
        if len(indices)<3: raise IndexError("Length of indices argument must be >=3.")
        if (indices[0]>=len(this_dataset) or indices[1]>=len(this_dataset)): raise IndexError("First or middle index cannot be greater than length of dataset.")
        if indices[0]>indices[1] or indices[1]>indices[2] or (len(indices)>3 and indices[2]>indices[3]): raise IndexError("Make sure indices are in ascending order left to right.")
    index = int(min(len(this_dataset),max_events)*split)
    train_indices = range(index) if indices is None else range(indices[0],int(min(len(this_dataset),indices[1])))
    train_dataset = Subset(this_dataset,train_indices)

    # Create training dataloader
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)

    # Load validation data
    # val_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    # val_dataset.load() #NOTE: DON'T Reload the entire dataset if you are splitting it anyway
    index2 = int(min(len(this_dataset),max_events))
    val_indices = range(index,index2) if indices is None else range(indices[1],int(min(len(this_dataset),indices[2])))
    val_dataset = Subset(this_dataset,val_indices)

    # Create testing dataloader
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)

    if indices is not None and len(indices)>=4:

        # Load validation data
        # val_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
        # val_dataset.load() #NOTE: DON'T Reload the entire dataset if you are splitting it anyway
        eval_indices = range(indices[2],last_index) if indices is None else range(indices[2],int(min(len(this_dataset),indices[3])))
        eval_dataset = Subset(this_dataset,eval_indices)

        # Create testing dataloader
        eval_loader = GraphDataLoader(
            eval_dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers)


        return train_loader, val_loader, eval_loader, num_labels, node_feature_dim, edge_feature_dim
    else:
        return train_loader, val_loader, num_labels, node_feature_dim, edge_feature_dim

def train(
    args,
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    max_epochs,
    dataset="",
    prefix="",
    log_interval=10,
    log_dir="logs/",
    save_path="model",
    verbose=True,
    **kwargs
    ):

    """
    Parameters
    ----------
    args : str, required
    model : str, required
    device : str, required
    train_loader : str, required
    val_loader : str, required
    optimizer : str, required
    scheduler : str, required
    criterion : str, required
    max_epochs : str, required
    dataset : str, optional
        Default : ""
    prefix : str, optional
        Default : ""
    log_interval : int, optional
        Default : 10.
    log_dir : str, optional
        Default : "logs/".
    save_path : str, optional
        Default : "model"
    verbose : bool, optional
        Default : True
    **kwargs

    Returns
    -------
    logs : dict
        Dictionary of training and validation metric lists organized by epoch
    
    Examples
    --------

    Notes
    -----

    """

    # Make sure log/save directories exist
    try:
        os.makedirs(log_dir+"/tb_logs/tmp") #NOTE: Do NOT use os.path.join() here since it requires that the directory exist.
    except Exception:
        if verbose: print("Could not create directory:",os.path.join(log_dir,"/tb_logs/tmp"))

    # # Make model parallel if training with multiple gpus
    # if device.type=='cuda' and device.index==None:
    #     model = DataParallel(model)

    # Show model if requested
    if verbose: print(model)

    figsize=(16,10)
    plt.rc('font', size=15) #controls default text size
    plt.rc('axes', titlesize=25) #fontsize of the title
    plt.rc('axes', labelsize=25) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=20) #fontsize of the y tick labels
    plt.rc('legend', fontsize=15) #fontsize of the legend

    # Logs for matplotlib plots
    logs={'train':{'loss':[],'accuracy':[],'roc_auc':[]}, 'val':{'loss':[],'accuracy':[],'roc_auc':[]}}

    use_modeloss=False #NOTE: ADDED 12/30/22
    mloss_coeff = 1.0
    modeloss = None
    if 'modeloss' in kwargs.keys() and kwargs['modeloss'] is not None:
        use_modeloss=True
        modeloss = kwargs['modeloss']
        
    if 'mloss_coeff' in kwargs.keys() and kwargs['mloss_coeff'] is not None:
        mloss_coeff = kwargs['mloss_coeff']

    my_mean = 1.08
    my_sig  = (1.24-1.08)/2 #NOTE: ADDED 1/9/23

    # Create train function
    def train_step(engine, batch):

        # Ensure model is in training mode
        model.train()

        # Get predictions and loss from data and labels
        x, label   = batch
        y = label[:,0].clone().detach().long() #NOTE: This assumes labels is 2D.
        x      = x.to(device)
        y      = y.to(device)
        y_pred = model(x)
        probs_Y = torch.softmax(y_pred, 1)
        y_pred = probs_Y
        loss   = criterion(probs_Y, y)

        m=None #NOTE: DEFAULT NECESSARY FOR USING WITHOUT MODELOSS
        if use_modeloss: #NOTE: ADDED 12/30/22
            m = label[:,1].clone().detach().float() #NOTE: This assumes labels is 2D. 
            m = m.to(device)
            m -= (my_mean+my_sig)
            m /= my_sig
            #NOTE: Check that there are enough background events for the modeloss computation
            mloss = mloss_coeff*(modeloss(y_pred, y, m) if (len(y)-y.count_nonzero()).item()>1 and len(m[y==modeloss.background_label])>=modeloss.bins else 0.0) 
            loss = mloss + loss

        # Step optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Apply softmax and get accuracy
        test_Y = y.clone().detach().float().view(-1, 1) 
        probs_Y = torch.softmax(y_pred, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
        acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)

        return {
                'y_pred': y_pred,
                'y': y,
                'm': m,
                'probs_y': probs_Y,
                'y_pred_preprocessed': argmax_Y,
                'loss': loss.detach().item(),
                'accuracy': acc
                }

    # Create validation function
    def val_step(engine, batch):

        # Ensure model is in evaluation mode
        model.eval()

        with torch.no_grad(): #NOTE: Important to call both model.eval and with torch.no_grad()! See https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval.

            # Get predictions and loss from data and labels
            x, label   = batch
            y = label[:,0].clone().detach().long() #NOTE: This assumes labels is 2D.
            x      = x.to(device)
            y      = y.to(device)
            y_pred = model(x)
            probs_Y = torch.softmax(y_pred, 1)
            y_pred = probs_Y
            loss   = criterion(probs_Y, y)
            
            m=None #NOTE: DEFAULT NECESSARY FOR USING WITHOUT MODELOSS
            if use_modeloss:
                m = label[:,1].clone().detach().float() #NOTE: This assumes labels is 2D.
                m = m.to(device)
                m -= (my_mean+my_sig)
                m /= my_sig
                #NOTE: Check that there are enough background events for the modeloss computation. 
                mloss = mloss_coeff*(modeloss(y_pred, y, m) if (len(y)-y.count_nonzero()).item()>1 and len(m[y==modeloss.background_label])>=modeloss.bins else 0.0) 
                loss = mloss + loss

            #------------ NOTE: NO BACKPROPAGATION FOR VALIDATION ----------#
            # # Step optimizer
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            #------------ NOTE: NO BACKPROPAGATION FOR VALIDATION ----------#

            # Apply softmax and get accuracy
            test_Y = y.clone().detach().float().view(-1, 1) 
            probs_Y = torch.softmax(y_pred, 1)
            argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
            acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)

        return {
                'y_pred': y_pred,
                'y': y,
                'm': m,
                'probs_y': probs_Y,
                'y_pred_preprocessed': argmax_Y,
                'loss': loss.detach().item(),
                'accuracy': acc
                }

    # Create trainer
    trainer = Engine(train_step)

    # Add training metrics
    accuracy  = Accuracy(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    accuracy.attach(trainer, 'accuracy')
    loss      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
    if use_modeloss:
        def myhelper_func(y_pred,y__,m__):
            return mloss_coeff*(modeloss(y_pred,y__,m__) if (len(y__)-y__.count_nonzero()).item()>1 and len(m__[y__==modeloss.background_label])>=modeloss.bins else 0.0)
        loss = Loss(lambda y_pred, y : myhelper_func(y_pred,y['y'],y['m'])+criterion(y_pred,y['y']), output_transform=lambda x: [x['y_pred'], {'y':x['y'], 'm':x['m']}])#NOTE: ADDED 12/30/22
    loss.attach(trainer, 'loss')
    # roc_auc   = ROC_AUC(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    # roc_auc.attach(trainer,'roc_auc')

    # Create evaluator
    evaluator = Engine(val_step)

    # Add evaluation metrics
    accuracy_  = Accuracy(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    accuracy_.attach(evaluator, 'accuracy')
    loss_      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
    if use_modeloss:
        def myhelper_func_(y_pred,y__,m__):
            return mloss_coeff*(modeloss(y_pred,y__,m__) if (len(y__)-y__.count_nonzero()).item()>1 and len(m__[y__==modeloss.background_label])>=modeloss.bins else 0.0)
        loss_ = Loss(lambda y_pred, y : myhelper_func_(y_pred,y['y'],y['m'])+criterion(y_pred,y['y']), output_transform=lambda x: [x['y_pred'], {'y':x['y'], 'm':x['m']}])#NOTE: ADDED 12/30/22 
    loss_.attach(evaluator, 'loss')
    # roc_auc_   = ROC_AUC(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    # roc_auc_.attach(evaluator,'roc_auc')
    
    # Set up early stopping
    def score_function(engine):
        val_loss = engine.state.metrics['loss']#DEBUGGING: COMMENTED OUT 1/9/23
        #val_loss = engine.state.output['loss']
        return -val_loss

    handler = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        cumulative_delta=args.cumulative_delta,
        score_function=score_function,
        trainer=trainer
        )
    evaluator.add_event_handler(Events.COMPLETED, handler) #NOTE: The handler is attached to an evaluator which runs one epoch on validation dataset.

    # Print training loss and accuracy
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def print_training_loss(trainer):
        if verbose: print(f"\rEpoch[{trainer.state.epoch}/{max_epochs} : " +
            f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
            f"Loss: {trainer.state.output['loss']:.3f} Accuracy: {trainer.state.output['accuracy']:.3f}",end='')

    # Step learning rate #NOTE: DEBUGGING: TODO: Replace above...
    @trainer.on(Events.EPOCH_COMPLETED)
    def stepLR(trainer):
        if type(scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(trainer.state.output['loss'])#TODO: NOTE: DEBUGGING.... Fix this...
        else:
            scheduler.step()

    # Log training metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(trainer):
        metrics = evaluator.run(train_loader).metrics
        for metric in metrics.keys(): logs['train'][metric].append(metrics[metric])
        if verbose: print(f"\nTraining Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['loss']:.4f} Avg accuracy: {metrics['accuracy']:.4f}")

    # Log validation metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(trainer):
        metrics = evaluator.run(val_loader).metrics
        for metric in metrics.keys(): logs['val'][metric].append(metrics[metric])
        if verbose: print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['loss']:.4f} Avg accuracy: {metrics['accuracy']:.4f}")

    """
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
        metric_names=["loss","accuracy"], #,"roc_auc" #OLD: 7/29/22
        global_step_transform=global_step_from_engine(trainer),
    )

    # Attach the logger to the evaluator on the validation dataset and log Loss, Accuracy metrics after
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["loss","accuracy"], #,"roc_auc" #OLD: 7/29/22
        global_step_transform=global_step_from_engine(evaluator)
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        param_name='lr'  # optional
    )
    """

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)
    ####tb_logger.close() #IMPORTANT! #NOTE: DEBUGGING COMMENTED OUT 1/9/23
    if save_path!="":
        torch.save(model.to('cpu').state_dict(), os.path.join(log_dir,save_path+"_weights")) #NOTE: Save to cpu state so you can test more easily.
        # torch.save(model.to('cpu'), os.path.join(log_dir,save_path)) #NOTE: Save to cpu state so you can test more easily.
   
    # Create training/validation loss plot
    f = plt.figure(figsize=figsize)
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
    f = plt.figure(figsize=figsize)
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['accuracy'],label="training")
    plt.plot(logs['val']['accuracy'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(0.0,1.0) #NOTE: ADDED 11/17/22
    f.savefig(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))

    # Write logs to json file #NOTE: ADDED 10/24/22
    with open(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.json'),'w') as fp:
        json.dump(logs,fp)

    return logs

def train_dagnn(
    args,
    model,
    classifier,
    discriminator,
    device,
    train_loader,
    val_loader,
    dom_train_loader,
    dom_val_loader,
    model_optimizer,
    classifier_optimizer,
    discriminator_optimizer,
    scheduler,
    train_criterion,
    dom_criterion,
    alpha,#TODO: Commented out for DEBUGGING
    max_epochs,
    dataset="",
    prefix="",
    log_interval=10,
    log_dir="logs/",
    save_path="model",
    verbose=True
    ):

    """
    Parameters
    ----------
    args : str, required #TODO: Fix these and get rid of args argument...
    model : str, required
    classifier : str, required
    discriminator : str, required
    device : str, required
    train_loader : str, required
    val_loader : str, required
    dom_train_loader : str, required
    dom_val_loader : str, required
    model_optimizer : str, required
    classifier_optimizer : str, required
    discriminator_optimizer : str, required
    scheduler : str, required
    train_criterion : str, required
    dom_criterion : str, required
    alpha : function, required
    max_epochs : int, required
    dataset : str, optional
        Default : ""
    prefix : str, optional
        Default : ""
    log_interval : int, optional
        Default : 10
    log_dir : str, optional
        Default : "logs/"
    save_path : str, optional
        Default : "model".
    verbose : bool, optional
        Default : True.

    Returns
    -------
    logs : dict
        Dictionary of training and validation metric lists organized by epoch

    Examples
    --------

    Notes
    -----

    """

    # Make sure log/save directories exist
    try:
        os.makedirs(log_dir+"/tb_logs/tmp") #NOTE: Do NOT use os.path.join() here since it requires that the directory exist.
    except Exception:
        if verbose: print("Could not create directory:",os.path.join(log_dir,"/tb_logs/tmp"))

    # # Make model parallel if training with multiple gpus
    # if device.type=='cuda' and device.index==None:
    #     model = DataParallel(model)

    # Show model if requested
    if verbose: print(model)

    figsize=(16,10)
    plt.rc('font', size=15) #controls default text size
    plt.rc('axes', titlesize=25) #fontsize of the title
    plt.rc('axes', labelsize=25) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=20) #fontsize of the y tick labels
    plt.rc('legend', fontsize=15) #fontsize of the legend 

    # Logs for matplotlib plots
    logs={'train':{'train_loss':[],'train_accuracy':[],'train_roc_auc':[],'dom_loss':[],'dom_accuracy':[],'dom_roc_auc':[]},
            'val':{'train_loss':[],'train_accuracy':[],'train_roc_auc':[],'dom_loss':[],'dom_accuracy':[],'dom_roc_auc':[]}}

    # Continuously sample target domain data for training and validation
    dom_train_set = itertools.cycle(dom_train_loader)
    dom_val_set   = itertools.cycle(dom_val_loader)

    # Create train function
    def train_step(engine, batch):

        # Ensure model is in training mode
        model.train()

        # Get domain data
        tgt = dom_train_set.__next__()[0] #NOTE: This returns [dgl.HeteroGraph,torch.tensor] for graph and labels.
        tgt = tgt.to(device)

        # Get predictions and loss from data and labels
        x, label     = batch
        train_labels = label[:,0].clone().detach().long() #NOTE: This assumes labels is 2D.
        x            = x.to(device)
        train_labels = train_labels.to(device)

        # Concatenate classification data and domain data
        x = dgl.unbatch(x)
        tgt = dgl.unbatch(tgt)
        nLabelled   = len(x)
        nUnlabelled = len(tgt)
        x.extend(tgt)

        del tgt #NOTE: CLEANUP STEP: DEBUGGING

        # # Move data to same device as model
        # x            = x.to(device)
        # train_labels = train_labels.to(device)

        # for el in tgt: #NOTE: Append test domain data since concatenation doesn't work.
        #     x.append(el)
        x = dgl.batch(x) #NOTE: Training and domain data must have the same schema for this to work.

        # Get hidden representation from model on training and domain data
        h = model(x)

        del x #NOTE: CLEANUP STEP: DEBUGGING
        
        # Step the domain discriminator on training and domain data
        dom_y = discriminator(h.detach())
        dom_labels = torch.cat([torch.ones(nLabelled,dtype=torch.long), torch.zeros(nUnlabelled,dtype=torch.long)], dim=0).to(device) #NOTE: Make sure domain label lengths match actual batches at the end.
        # dom_labels = torch.cat([torch.cat([torch.ones(nLabelled,1),torch.zeros(nLabelled,1)],dim=1),torch.cat([torch.zeros(nUnlabelled,1),torch.ones(nUnlabelled,1)],dim=1)],dim=0).to(device)
        dom_loss = dom_criterion(dom_y, dom_labels) #NOTE: Using activation function like nn.Sigmoid() at end of model is important since the predictions need to be in [0,1].
        discriminator.zero_grad()
        dom_loss.backward()
        discriminator_optimizer.step()
        
        # Step the classifier on training data
        train_y = classifier(h[:nLabelled]) #NOTE: Only train on labelled (i.e., training) data, not domain data.
        dom_y = discriminator(h)
        del h #NOTE: CLEANUP STEP: DEBUGGING
        train_loss = train_criterion(train_y, train_labels)
        dom_loss   = dom_criterion(dom_y, dom_labels) #NOTE: Using nn.Sigmoid() is important since the predictions need to be in [0,1].

        # Get total loss using lambda coefficient for epoch
        # coeff = alpha(engine.state.epoch, max_epochs)#OLD: 7/25/22
        tot_loss = train_loss - alpha * dom_loss
        
        # Zero gradients in all parts of model
        model.zero_grad()
        classifier.zero_grad()
        discriminator.zero_grad()
        
        # Step total loss
        tot_loss.backward()
        
        # Step classifier and model optimizers (backwards)
        classifier_optimizer.step()
        model_optimizer.step()

        # Apply softmax and get accuracy on training data
        train_true_y = train_labels.clone().detach().float().view(-1, 1) #NOTE: Labels for cross entropy loss have to be (N) shaped if input is (N,C) shaped.
        train_probs_y = torch.softmax(train_y, 1)
        train_argmax_y = torch.max(train_probs_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
        train_acc = (train_true_y == train_argmax_y.float()).sum().item() / len(train_true_y)

        # Apply softmax and get accuracy on domain data
        dom_true_y = dom_labels.clone().detach().float().view(-1, 1)
        dom_probs_y = torch.softmax(dom_y, 1) #NOTE: Activation should already be a part of the discriminator #NOTE CHANGED BACK TO ACTUALLY USE THIS STEP 11/17/22
        dom_argmax_y = torch.max(dom_probs_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
        dom_acc = (dom_true_y == dom_argmax_y.float()).sum().item() / len(dom_true_y)

        return {
                'alpha': alpha,
                'train_y': train_y, #CLASSIFIER OUTPUT
                'train_probs_y': train_probs_y,
                'train_true_y': train_labels, #NOTE: Need this for some reason?
                'train_argmax_y': train_argmax_y,
                'train_loss': train_loss.detach().item(),
                'train_accuracy': train_acc,
                'dom_y': dom_y, #DISCRIMINATOR OUTPUT
                'dom_true_y': dom_labels, #NOTE: Need this for some reason?
                'dom_argmax_y': dom_argmax_y,
                'dom_loss': dom_loss.detach().item(),
                'dom_accuracy': dom_acc,
                'tot_loss': tot_loss.detach().item() #TOTAL LOSS
                }

    # Create validation function
    def val_step(engine, batch):

        # Ensure model is in evaluation mode
        model.eval()

        with torch.no_grad(): #NOTE: Important to call both model.eval and with torch.no_grad()! See https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval.
            
            # Get domain data
            tgt = dom_val_set.__next__()[0] #NOTE: This returns [dgl.HeteroGraph,torch.tensor] for graph and labels.
            tgt = tgt.to(device)

            # Get predictions and loss from data and labels
            x, label     = batch
            train_labels = label[:,0].clone().detach().long() #NOTE: This assumes labels is 2D.
            del label #NOTE: CLEANUP STEP: DEBUGGING
            x            = x.to(device)
            train_labels = train_labels.to(device)

            # Concatenate classification data and domain data
            x = dgl.unbatch(x)
            tgt = dgl.unbatch(tgt)
            nLabelled   = len(x)
            nUnlabelled = len(tgt)
            x.extend(tgt)

            del tgt #NOTE: CLEANUP STEP: DEBUGGING

            # # Move data to same device as model
            # x            = x.to(device)
            # train_labels = train_labels.to(device)

            # for el in tgt: #NOTE: Append test domain data since concatenation doesn't work.
            #     x.append(el)
            x = dgl.batch(x) #NOTE: Training and domain data must have the same schema for this to work.

            # Get hidden representation from model on training and domain data
            h = model(x)

            del x #NOTE: CLEANUP STEP: DEBUGGING
            
            # Step the domain discriminator on training and domain data
            dom_y = discriminator(h.detach())
            dom_labels = torch.cat([torch.ones(nLabelled,dtype=torch.long), torch.zeros(nUnlabelled,dtype=torch.long)], dim=0).to(device) #NOTE: Make sure domain label lengths match actual batches at the end.
            # dom_labels = torch.cat([torch.cat([torch.ones(nLabelled,1),torch.zeros(nLabelled,1)],dim=1),torch.cat([torch.zeros(nUnlabelled,1),torch.ones(nUnlabelled,1)],dim=1)],dim=0).to(device)
            dom_loss = dom_criterion(dom_y, dom_labels) #NOTE: Using activation function like nn.Sigmoid() at end of model is important since the predictions need to be in [0,1].

            #------------ NOTE: NO BACKPROPAGATION FOR VALIDATION ----------#
            # discriminator.zero_grad()
            # dom_loss.backward()
            # discriminator_optimizer.step()
            #------------ NOTE: NO BACKPROPAGATION FOR VALIDATION ----------#
            
            # Step the classifier on training data
            train_y = classifier(h[:nLabelled]) #NOTE: Only evaluate on labelled (i.e., training) data, not domain data.
            dom_y = discriminator(h)
            del h #NOTE: CLEANUP STEP: DEBUGGING
            train_loss = train_criterion(train_y, train_labels)
            dom_loss   = dom_criterion(dom_y, dom_labels) #NOTE: Using activation like nn.Sigmoid() on discriminator is important since the predictions need to be in [0,1].

            # Get total loss using lambda coefficient for epoch
            # coeff = alpha(engine.state.epoch, max_epochs)#OLD: 7/25/22
            tot_loss = train_loss - alpha * dom_loss
            
            #------------ NOTE: NO BACKPROPAGATION FOR VALIDATION ----------#
            # # Zero gradients in all parts of model
            # model.zero_grad()
            # classifier.zero_grad()
            # discriminator.zero_grad()
            
            # # Step total loss
            # tot_loss.backward()
            
            # # Step classifier and model optimizers (backwards)
            # classifier_optimizer.step()
            # model_optimizer.step()
            #------------ NOTE: NO BACKPROPAGATION FOR VALIDATION ----------#

            # Apply softmax and get accuracy on training data
            train_true_y = train_labels.clone().detach().float().view(-1, 1) #NOTE: Labels for cross entropy loss have to be (N) shaped if input is (N,C) shaped.
            train_probs_y = torch.softmax(train_y, 1)
            train_argmax_y = torch.max(train_probs_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
            train_acc = (train_true_y == train_argmax_y.float()).sum().item() / len(train_true_y)

            # Apply softmax and get accuracy on domain data
            dom_true_y = dom_labels.clone().detach().float().view(-1, 1)
            dom_probs_y = torch.softmax(dom_y, 1) #NOTE: Activation should already be a part of the discriminator #NOTE: CHANGED 10/24/22 added this step back in and now using dom_probs_y below ... not sure if this will work
            dom_argmax_y = torch.max(dom_probs_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
            dom_acc = (dom_true_y == dom_argmax_y.float()).sum().item() / len(dom_true_y)

        return {
                'alpha': alpha,
                'train_y': train_y, #CLASSIFIER OUTPUT
                'train_probs_y': train_probs_y,
                'train_true_y': train_labels, #NOTE: Need this for some reason?
                'train_argmax_y': train_argmax_y,
                'train_loss': train_loss.detach().item(),
                'train_accuracy': train_acc,
                'dom_y': dom_y, #DISCRIMINATOR OUTPUT
                'dom_true_y': dom_labels, #NOTE: Need this for some reason?
                'dom_argmax_y': dom_argmax_y,
                'dom_loss': dom_loss.detach().item(),
                'dom_accuracy': dom_acc,
                'tot_loss': tot_loss.detach().item() #TOTAL LOSS
                }

    # Create trainer
    trainer = Engine(train_step)

    # Add training metrics for classifier
    train_accuracy  = Accuracy(output_transform=lambda x: [x['train_probs_y'], x['train_true_y']])
    train_accuracy.attach(trainer, 'train_accuracy')
    train_loss      = Loss(train_criterion,output_transform=lambda x: [x['train_y'], x['train_true_y']])
    train_loss.attach(trainer, 'train_loss')
    # train_roc_auc   = ROC_AUC(output_transform=lambda x: [x['train_probs_y'], x['train_true_y']]) #NOTE: ROC_AUC CURRENTLY NOT WORKING HERE, NOT SURE WHY...
    # train_roc_auc.attach(trainer,'train_roc_auc')

    # Add training metrics for discriminator
    dom_accuracy  = Accuracy(output_transform=lambda x: [x['dom_argmax_y'], x['dom_true_y']])
    dom_accuracy.attach(trainer, 'dom_accuracy')
    dom_loss      = Loss(dom_criterion,output_transform=lambda x: [x['dom_y'], x['dom_true_y']])
    dom_loss.attach(trainer, 'dom_loss')
    # dom_roc_auc   = ROC_AUC(output_transform=lambda x: [x['dom_argmax_y'], x['dom_true_y']]) #NOTE: ROC_AUC CURRENTLY NOT WORKING HERE, NOT SURE WHY...
    # dom_roc_auc.attach(trainer,'dom_roc_auc')

    # Create evaluator
    evaluator = Engine(val_step)

    # Add validation metrics for classifier
    _train_accuracy  = Accuracy(output_transform=lambda x: [x['train_probs_y'], x['train_true_y']])
    _train_accuracy.attach(evaluator, 'train_accuracy')
    _train_loss      = Loss(train_criterion,output_transform=lambda x: [x['train_y'], x['train_true_y']])
    _train_loss.attach(evaluator, 'train_loss')
    # _train_roc_auc   = ROC_AUC(output_transform=lambda x: [x['train_probs_y'], x['train_true_y']]) #NOTE: ROC_AUC CURRENTLY NOT WORKING HERE, NOT SURE WHY...
    # _train_roc_auc.attach(evaluator,'train_roc_auc')

    # Add validation metrics for discriminator
    _dom_accuracy  = Accuracy(output_transform=lambda x: [x['dom_argmax_y'], x['dom_true_y']])
    _dom_accuracy.attach(evaluator, 'dom_accuracy')
    _dom_loss      = Loss(dom_criterion,output_transform=lambda x: [x['dom_y'], x['dom_true_y']])
    _dom_loss.attach(evaluator, 'dom_loss')
    # _dom_roc_auc   = ROC_AUC(output_transform=lambda x: [x['dom_argmax_y'], x['dom_true_y']]) #NOTE: ROC_AUC CURRENTLY NOT WORKING HERE, NOT SURE WHY...
    # _dom_roc_auc.attach(evaluator,'dom_roc_auc')

    # Set up early stopping
    def score_function(engine):
        val_loss = engine.state.metrics['train_loss']
        return -val_loss

    handler = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        cumulative_delta=args.cumulative_delta,
        score_function=score_function,
        trainer=trainer
        )
    evaluator.add_event_handler(Events.COMPLETED, handler) #NOTE: The handler is attached to an evaluator which runs one epoch on validation dataset.

    # Print training loss and accuracy
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def print_training_loss(trainer):
        if verbose: print(
            f"\rEpoch[{trainer.state.epoch}/{max_epochs} : " +
            f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
            f"Classifier Loss: {trainer.state.output['train_loss']:.3f} Accuracy: {trainer.state.output['train_accuracy']:.3f} " +
            f"Discriminator: Loss: {trainer.state.output['dom_loss']:.3f} Accuracy: {trainer.state.output['dom_accuracy']:.3f}",
            end='')

    # Step learning rate
    @trainer.on(Events.EPOCH_COMPLETED)
    def stepLR(trainer):
        if type(scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(trainer.state.output['train_loss'])#TODO: NOTE: DEBUGGING.... Fix this...
        else:
            scheduler.step()

    # Log training metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(trainer):
        metrics = evaluator.run(train_loader).metrics
        for metric in metrics.keys(): logs['train'][metric].append(metrics[metric])
        if verbose: print(
            f"\nTraining Results - Epoch: {trainer.state.epoch} Classifier: loss: {metrics['train_loss']:.4f} accuracy: {metrics['train_accuracy']:.4f} Discriminator: loss: {metrics['dom_loss']:.4f} accuracy: {metrics['dom_accuracy']:.4f}")

    # Log validation metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(trainer):
        metrics = evaluator.run(val_loader).metrics
        for metric in metrics.keys(): logs['val'][metric].append(metrics[metric])
        if verbose: print(
            f"Validation Results - Epoch: {trainer.state.epoch}  Classifier loss: {metrics['train_loss']:.4f} accuracy: {metrics['train_accuracy']:.4f} Discriminator: loss: {metrics['dom_loss']:.4f} accuracy: {metrics['dom_accuracy']:.4f}")

    # Create a TensorBoard logger
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training_by_iteration",
        output_transform=lambda x: x["train_loss"]
    )
        
    # Attach the logger to the evaluator on the training dataset and log Loss, Accuracy metrics after each epoch
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=["train_loss","train_accuracy","dom_loss","dom_accuracy"],
        # metric_names=["train_loss","train_accuracy","train_roc_auc","dom_loss","dom_accuracy","dom_roc_auc"], #NOTE: TODO: OLD
        global_step_transform=global_step_from_engine(trainer),
    )

    # Attach the logger to the evaluator on the validation dataset and log Loss, Accuracy metrics after
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["train_loss","train_accuracy","dom_loss","dom_accuracy"],
        # metric_names=["train_loss","train_accuracy","train_roc_auc","dom_loss","dom_accuracy","dom_roc_auc"], #NOTE: TODO: OLD
        global_step_transform=global_step_from_engine(evaluator)
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=model_optimizer,
        param_name='lr'  # optional
    )#TODO: Add other learning rates?

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)
    tb_logger.close() #IMPORTANT!
    if save_path!="":
        torch.save(model.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_model_weights')) #NOTE: Save to cpu state so you can test more easily.
        torch.save(classifier.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_classifier_weights'))
        torch.save(discriminator.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_discriminator_weights'))
        # torch.save(model.to('cpu'), os.path.join(log_dir,save_path+'_model')) #NOTE: Save to cpu state so you can test more easily.
        # torch.save(classifier.to('cpu'), os.path.join(log_dir,save_path+'_classifier'))
        # torch.save(discriminator.to('cpu'), os.path.join(log_dir,save_path+'_discriminator'))

    # Create training/validation loss plot
    f = plt.figure(figsize=figsize)
    plt.subplot()
    plt.title('Loss per epoch')
    plt.plot(logs['train']['train_loss'],'-',color='orange',label="classifier training")
    plt.plot(logs['val']['train_loss'],'-',color='red',label="classifier validation")
    plt.plot(logs['train']['dom_loss'],'--',color='orange',label="discriminator training")
    plt.plot(logs['val']['dom_loss'],'--',color='red',label="discriminator validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    f.savefig(os.path.join(log_dir,'training_metrics_loss_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))

    # Create training/validation accuracy plot
    f = plt.figure(figsize=figsize)
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['train_accuracy'],'-',color='blue',label="classifier training")
    plt.plot(logs['val']['train_accuracy'],'-',color='purple',label="classifier validation")
    plt.plot(logs['train']['dom_accuracy'],'--',color='blue',label="discriminator training")
    plt.plot(logs['val']['dom_accuracy'],'--',color='purple',label="discriminator validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(0.0,1.0)#NOTE ADDED 11/17/22
    f.savefig(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))

    # Write logs to json file #NOTE: ADDED 10/24/22
    with open(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.json'),'w') as fp:
        json.dump(logs,fp)

    return logs
    
def evaluate(model,device,eval_loader=None,dataset="", prefix="", split=1.0, max_events=1e20 , log_dir="logs/",verbose=True, roc_cut=None, model1=None, use_umap=True):
    #TODO: Update args defaults for split and max_events!

    #TODO: Make these options...
    plt.rc('font', size=15) #controls default text size                                                                                                                     
    plt.rc('axes', titlesize=25) #fontsize of the title                                                                                                                     
    plt.rc('axes', labelsize=25) #fontsize of the x and y labels                                                                                                            
    plt.rc('xtick', labelsize=20) #fontsize of the x tick labels                                                                                                            
    plt.rc('ytick', labelsize=20) #fontsize of the y tick labels                                                                                                            
    plt.rc('legend', fontsize=15) #fontsize of the legend

    figsize = (16,10) #NOTE: ADDED 9/30/22
    
    # Load validation data
    test_dataset = GraphDataset(prefix+dataset) if eval_loader is None else eval_loader.dataloader.dataset # Make sure this is copied into ~/.dgl folder
    if eval_loader is None:
        test_dataset.load()
        test_dataset = Subset(test_dataset,range(int(min(len(test_dataset),max_events)*split)))

    model.eval()
    model      = model.to(device)

    test_bg    = dgl.batch(test_dataset.dataset.graphs[test_dataset.indices.start:test_dataset.indices.stop]) #TODO: Figure out nicer way to use subset
    test_Y     = test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float().view(-1, 1) #IMPORTANT: keep .view() here
    test_bg    = test_bg.to(device)
    test_Y     = test_Y.to(device)

    print("DEBUGGING: test_Y.device = ",test_Y.device)#DEBUGGING
    #----- ADDED -----#
    print("DEBUGGING: test_bg.device = ",test_bg.device)#DEBUGGING
    #print("DEBUGGING: model.models = ",model.models)#DEBUGGING
    try:
        print("DEBUGGING: model.models = ",model.models)#DEBUGGING
    except Exception:
        print("DEBUGGING: could not access attribute model.models")#DEBUGGING
    try:
        print("DEBUGGING: model.models[0].device = ",model.models[0].device)#DEBUGGING
    except Exception:
        print("DEBUGGING: could not access attribute model.models[0].device")#DEBUGGING
    try:
        model.models = [model.models[idx].to(device) for idx in range(len(model.models))]#DEBUGGING
    except Exception:
        print("DEBUGGING: model.models = [model.models[idx].to(device) for idx in range(len(model.models))]")#DEBUGGING
    #----- ADDED END -----#  

    prediction = model(test_bg)
    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1) if roc_cut is None else torch.tensor([1 if el>roc_cut else 0 for el in probs_Y[:,1]],dtype=torch.long) #NOTE: ADDED 10/25/22
    test_acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
    if verbose: print('Accuracy of predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
    
    """
    #try: #DEBUGGING #ADDED 11/29/22
    import umap
    import seaborn as sns
    reducer = umap.UMAP()
    extractor = model.models[0]
    latent_repr = extractor(test_bg).detach()
    print("DEBUGGING: type(latent_repr) = ",type(latent_repr))#DEBUGGING
    print("DEBUGGING: latent_repr = ",latent_repr)#DEBUGGING
    embedding = reducer.fit_transform(latent_repr)
    print("DEBUGGING: type(embedding) = ",type(embedding))#DEBUGGING
    print("DEBUGGING: embedding.shape = ",embedding.shape)#DEBUGGING
    f = plt.figure((16,10))
    plt.scatter(embedding[:,0],embedding[:,1])
    plt.title('UMAP projection of the DAGIN latent space presentation')
    f.savefig(os.path.join(log_dir,'umap_basic.png'))

    # Plot separated into true/false sig/bg
    
    #except Exception:
    #    print("COULD NOT IMPORT UMAP OR ACCESS MODEL EXTRACTOR ELEMENT...")#DEBUGGING ADDED 11/29/22
    """

    #NOTE: ADDED 11/10/22
    if model1 is not None:
        model1.eval()
        model1      = model1.to(device)
        test_bg1    = dgl.batch(test_dataset.dataset.graphs[test_dataset.indices.start:test_dataset.indices.stop])#TODO: Figure out nicer way to use subset
        test_bg1    = test_bg1.to(device)
        prediction1 = model1(test_bg1)
        probs_Y1    = torch.softmax(prediction1, 1)
        argmax_Y1   = torch.max(probs_Y1, 1)[1].view(-1, 1) if roc_cut is None else torch.tensor([1 if el>roc_cut else 0 for el in probs_Y[:,1]],dtype=torch.long) #NOTE: ADDED 10/25/22
        argmax_Y    = torch.min(argmax_Y,argmax_Y1) if roc_cut is None else torch.tensor([1 if el>roc_cut and probs_Y1[:,1][idx]>roc_cut else 0 for idx, el in enumerate(probs_Y[:,1])],dtype=torch.long)

    # Copy arrays back to CPU
    test_Y   = test_Y.cpu()
    probs_Y  = probs_Y.cpu()
    argmax_Y = argmax_Y.cpu()
    prediction = prediction.cpu()#NOTE: ADDED 12/30/22

    # Get separated mass distributions
    mass_sig_Y    = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),mask=~(argmax_Y == 1)) #if eval_loader is None else ma.array(test_dataset.labels[:,0].clone().detach().float().view(-1,1),mask=~(argmax_Y == 1)) 
    mass_bg_Y     = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),mask=~(argmax_Y == 0)) #if eval_loader is None else ma.array(test_dataset.labels[:,0].clone().detach().float().view(-1,1),mask=~(argmax_Y == 0)) 

    # Get false-positive true-negatives and vice versa
    mass_sig_true  = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),
                                mask=np.logical_or(~(torch.squeeze(argmax_Y) == 1),~(torch.squeeze(argmax_Y) == test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float())))
    mass_bg_true   = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),
                                mask=np.logical_or(~(torch.squeeze(argmax_Y) == 0),~(torch.squeeze(argmax_Y) == test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float())))
    mass_sig_false = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),
                                mask=np.logical_or(~(torch.squeeze(argmax_Y) == 1),~(torch.squeeze(argmax_Y) != test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float())))
    mass_bg_false  = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),
                                mask=np.logical_or(~(torch.squeeze(argmax_Y) == 0),~(torch.squeeze(argmax_Y) != test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float())))

    # Get separated mass distributions MC-Matched
    mass_sig_MC   = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),mask=~(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0] == 1))
    mass_bg_MC    = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),mask=~(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0] == 0))

    np.save(log_dir+'mass_sig_Y_mask.npy',mass_sig_Y.mask)#NOTE: ADDED 12/1/22
    np.save(log_dir+'mass_bg_Y_mask.npy',mass_bg_Y.mask)#NOTE: ADDED 12/1/22
    np.save(log_dir+'mass_sig_true_mask.npy',mass_sig_true.mask)#NOTE: ADDED 12/1/22
    np.save(log_dir+'mass_bg_true_mask.npy',mass_bg_true.mask)#NOTE: ADDED 12/1/22
    np.save(log_dir+'mass_sig_false_mask.npy',mass_sig_false.mask)#NOTE: ADDED 12/1/22
    np.save(log_dir+'mass_bg_false_mask.npy',mass_bg_false.mask)#NOTE: ADDED 12/1/22
    np.save(log_dir+'mass_sig_MC_mask.npy',mass_sig_MC.mask)#NOTE: ADDED 12/1/22
    np.save(log_dir+'mass_bg_MC_mask.npy',mass_bg_MC.mask)#NOTE: ADDED 12/1/22

    #try: #DEBUGGING #ADDED 11/29/22
    import umap
    import seaborn as sns
    from sklearn.manifold import TSNE
    reducer = umap.UMAP() if use_umap else TSNE(2)
    extractor = None
    latent_repr = None
    try:
        extractor = model.models[0]
        latent_repr = extractor.get_latent_repr(test_bg).detach().cpu()#NOTE: ADDED .cpu() 12/30/22
        print("DEBUGGING: type(latent_repr) = ",type(latent_repr[0]))#DEBUGGING
        print("DEBUGGING: latent_repr[0] = ",latent_repr[0])#DEBUGGING
        print("DEBUGGING: len(latent_repr) = ",len(latent_repr))#DEBUGGING
    except AttributeError as ae:
        print("AttributeError: could not access model.models")#DEBUGGING
        print(ae)
        extractor = model
        latent_repr = extractor.get_latent_repr(test_bg).detach().cpu()#NOTE: ADDED .cpu() 12/30/22 
        print("DEBUGGING: type(latent_repr) = ",type(latent_repr[0]))#DEBUGGING
        print("DEBUGGING: latent_repr[0] = ",latent_repr[0])#DEBUGGING
        print("DEBUGGING: len(latent_repr) = ",len(latent_repr))#DEBUGGING
    print("DEBUGGING: type(latent_repr) = ",type(latent_repr))#DEBUGGING
    print("DEBUGGING: latent_repr = ",latent_repr)#DEBUGGING
    embedding = reducer.fit_transform(latent_repr)
    print("DEBUGGING: type(embedding) = ",type(embedding))#DEBUGGING
    print("DEBUGGING: embedding.shape = ",embedding.shape)#DEBUGGING 

    visualization_method = 'umap' if use_umap else 'tsne'

    np.save(log_dir+'latent_repr.npy',np.array(latent_repr))#DEBUGGING: NOTE ADDED 11/30/22
    np.save(log_dir+visualization_method+'_embedding.npy',np.array(embedding))#DEBUGGING: NOTE ADDED 11/30/22

    f = plt.figure(figsize=figsize)
    plt.scatter(embedding[:,0],embedding[:,1])
    plt.title(visualization_method+' projection of latent space representation')
    f.savefig(os.path.join(log_dir,visualization_method+'_basic.png'))
    # Plot separated into true/false sig/bg

    em_sig = ma.array(embedding,mask=[[el,el] for el in mass_sig_Y.mask])
    em_bg = ma.array(embedding,mask=[[el,el] for el in mass_bg_Y.mask])

    em_sig_true = ma.array(embedding,mask=[[el,el] for el in mass_sig_true.mask])
    em_bg_true = ma.array(embedding,mask=[[el,el] for el in mass_bg_true.mask])
    em_sig_false = ma.array(embedding,mask=[[el,el] for el in mass_sig_false.mask])
    em_bg_false = ma.array(embedding,mask=[[el,el] for el in mass_bg_false.mask])

    em_sig_MC = ma.array(embedding,mask=[[el,el] for el in mass_sig_MC.mask])
    em_bg_MC = ma.array(embedding,mask=[[el,el] for el in mass_bg_MC.mask])

    # Set plotting options
    alpha = 0.5
    mew   = 0.0

    # Plot visualization method decisions separated into signal/background
    f, ax = plt.subplots(figsize=figsize)
    plt.title('Separated '+visualization_method+' distribution')
    ax.scatter(em_sig[:,0][~em_sig.mask[:,0]],em_sig[:,1][~em_sig.mask[:,1]], alpha=alpha, linewidths=mew, label='sig')
    ax.scatter(em_bg[:,0][~em_bg.mask[:,0]],em_bg[:,1][~em_bg.mask[:,1]], alpha=alpha, linewidths=mew, label='bg')
    ax.legend(loc='upper left', frameon=False)
    f.savefig(os.path.join(log_dir,visualization_method+'_nn_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot visualization method decisions just signal
    f, ax = plt.subplots(figsize=figsize)
    plt.title(visualization_method+' sig distribution')
    ax.scatter(em_sig[:,0][~em_sig.mask[:,0]],em_sig[:,1][~em_sig.mask[:,1]], color="tab:blue", alpha=alpha, linewidths=mew, label='sig')
    #ax.scatter(em_bg[:,0][~em_bg.mask[:,0]],em_bg[:,1][~em_bg.mask[:,1]], alpha=alpha, linewidths=mew, label='bg')
    ax.legend(loc='upper left', frameon=False)
    f.savefig(os.path.join(log_dir,visualization_method+'_nn_sig_only_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot visualization method decisions just background
    f, ax = plt.subplots(figsize=figsize)
    plt.title(visualization_method+' bg distribution')
    #ax.scatter(em_sig[:,0][~em_sig.mask[:,0]],em_sig[:,1][~em_sig.mask[:,1]], alpha=alpha, linewidths=mew, label='sig')
    ax.scatter(em_bg[:,0][~em_bg.mask[:,0]],em_bg[:,1][~em_bg.mask[:,1]], color="tab:orange", alpha=alpha, linewidths=mew, label='bg')
    ax.legend(loc='upper left', frameon=False)
    f.savefig(os.path.join(log_dir,visualization_method+'_nn_bg_only_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot correct visualization method decisions separated into signal/background
    f, ax = plt.subplots(figsize=figsize)
    plt.title('Separated '+visualization_method+' distribution (true)')
    ax.scatter(em_sig_true[:,0][~em_sig_true.mask[:,0]],em_sig_true[:,1][~em_sig_true.mask[:,1]], alpha=alpha, linewidths=mew, label='sig')
    ax.scatter(em_bg_true[:,0][~em_bg_true.mask[:,0]],em_bg_true[:,1][~em_bg_true.mask[:,1]], alpha=alpha, linewidths=mew, label='bg')
    ax.legend(loc='upper left', frameon=False)
    f.savefig(os.path.join(log_dir,visualization_method+'_mc_matched_nn_true_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot incorrect visualization method decisions separated into signal/background
    f, ax = plt.subplots(figsize=figsize)
    plt.title('Separated '+visualization_method+' distribution (false)')
    ax.scatter(em_sig_false[:,0][~em_sig_false.mask[:,0]],em_sig_false[:,1][~em_sig_false.mask[:,1]], alpha=alpha, linewidths=mew, label='sig')
    ax.scatter(em_bg_false[:,0][~em_bg_false.mask[:,0]],em_bg_false[:,1][~em_bg_false.mask[:,1]], alpha=alpha, linewidths=mew, label='bg')
    ax.legend(loc='upper left', frameon=False)
    f.savefig(os.path.join(log_dir,visualization_method+'_mc_matched_nn_false_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot MC-Matched visualization method distributions
    f, ax = plt.subplots(figsize=figsize)
    plt.title('Separated'+visualization_method+' distribution MC-matched')
    ax.scatter(em_sig_MC[:,0][~em_sig_MC.mask[:,0]],em_sig_MC[:,1][~em_sig_MC.mask[:,1]], alpha=alpha, linewidths=mew, label='sig')
    ax.scatter(em_bg_MC[:,0][~em_bg_MC.mask[:,0]],em_bg_MC[:,1][~em_bg_MC.mask[:,1]], alpha=alpha, linewidths=mew, label='bg')
    ax.legend(loc='upper left', frameon=False)
    f.savefig(os.path.join(log_dir,visualization_method+'_mc_matched_nn_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot MC-Matched visualization method distributions for NN-identified signal
    f, ax = plt.subplots(figsize=figsize)
    plt.title('NN-identified signal '+visualization_method+' distribution MC-matched')
    ax.scatter(em_sig_true[:,0][~em_sig_true.mask[:,0]],em_sig_true[:,1][~em_sig_true.mask[:,1]], alpha=alpha, linewidths=mew, label='true')
    ax.scatter(em_sig_false[:,0][~em_sig_false.mask[:,0]],em_sig_false[:,1][~em_sig_false.mask[:,1]], alpha=alpha, linewidths=mew, label='false')
    ax.legend(loc='upper left', frameon=False)
    f.savefig(os.path.join(log_dir,visualization_method+'_mc_matched_nn_sig_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot MC-Matched visualization method distributions for NN-identified background
    f, ax = plt.subplots(figsize=figsize)
    plt.title('NN-identified bg '+visualization_method+' distribution MC-matched')
    ax.scatter(em_bg_true[:,0][~em_bg_true.mask[:,0]],em_bg_true[:,1][~em_bg_true.mask[:,1]], alpha=alpha, linewidths=mew, label='true')
    ax.scatter(em_bg_false[:,0][~em_bg_false.mask[:,0]],em_bg_false[:,1][~em_bg_false.mask[:,1]], alpha=alpha, linewidths=mew, label='false')
    ax.legend(loc='upper left', frameon=False)
    f.savefig(os.path.join(log_dir,visualization_method+'_mc_matched_nn_bg_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    #except Exception:
    #    print("COULD NOT IMPORT UMAP OR ACCESS MODEL EXTRACTOR ELEMENT...")#DEBUGGING ADDED 11/29/22

    
    ###################################################
    ## Define fit function
    #def func(x, N, beta, m, loc, scale, A, B, C):
    #    return N*crystalball.pdf(-x, beta, m, -loc, scale) + A*(1 - B*(x - C)**2)
    #    
    #def sig(x, N, beta, m, loc, scale):
    #    return N*crystalball.pdf(-x, beta, m, -loc, scale)
    #    
    #def bg(x, A, B, C):
    #    return A*(1 - B*(x - C)**2)
    ###################################################

    # # Set font sizes
    # plt.rc('axes', titlesize=30)
    # plt.rc('axes', labelsize=30)
    # plt.rc('xtick', labelsize=24)
    # plt.rc('ytick', labelsize=24)
    # plt.rc('legend', fontsize=24)

    print("DEBUGGING: np.min(mass_sig_Y) = ",np.min(mass_sig_Y))#DEBUGGING
    print("DEBUGGING: np.max(mass_sig_Y) = ",np.max(mass_sig_Y))#DEBUGGING

    # Plot mass decisions separated into signal/background
    bins = 100
    low_high = (1.08,1.24)#(1.1,1.13)
    f = plt.figure(figsize=(16,10))
    plt.title('Separated mass distribution')
    hdata = plt.hist(mass_sig_Y[~mass_sig_Y.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    
    # Fit output of NN
    print("DEBUGGING: hdata[0] = ",hdata[0])#DEBUGGING
    N, beta, m, loc, scale, A, B, C = 10, 1, 1.112, 1.115, 0.008, np.max(hdata[0][-10:-1]), 37, 1.24 #OLD: A = hdata[0][-1] #DEBUGGING COMMENTED OUT!
    if A == 0: A = 0.1 #DEBUGGING
    d_N, d_beta, d_m, d_loc, d_scale, d_A, d_B, d_C = N/0.01, beta/0.1, m/0.1, loc/0.1, scale/0.01, A/10, B/0.1, C/1
    parsMin = [N-d_N, beta-d_beta, m-d_m, loc-d_loc, scale-d_scale, B-d_B]#[N-d_N, beta-d_beta, m-d_m, loc-d_loc, scale-d_scale, A-d_A, B-d_B, C-d_C] #NOTE: CHANGED 10/21/22
    parsMax = [N+d_N, beta+d_beta, m+d_m, loc+d_loc, scale+d_scale, B+d_B]#[N+d_N, beta+d_beta, m+d_m, loc+d_loc, scale+d_scale, A+d_A, B+d_B, C+d_C]#NOTE: CHANGED 10/21/22
    print("DEBUGGING: parsMin = ",parsMin)#DEBUGGING
    print("DEBUGGING: parsMax = ",parsMax)#DEBUGGING

    ################################################# #NOTE: UPDATED FUNCTION DEFINITIONS 10/21/22
    # Define fit function
    def func(x, N, beta, m, loc, scale, B, A = A, C=C):
        return N*crystalball.pdf(-x, beta, m, -loc, scale) + A*(1 - B*(x - C)**2)

    def sig(x, N, beta, m, loc, scale):
        return N*crystalball.pdf(-x, beta, m, -loc, scale)

    def bg(x, B, A=A, C=C):
        return A*(1 - B*(x - C)**2)
    ##################################################
    
    optParams, pcov = opt.curve_fit(func, hdata[1][:-1], hdata[0], method='trf', bounds=(parsMin,parsMax))

    print("DEBUGGING: np.shape(pcov) = ",np.shape(pcov))#DEBUGGING 9/27/22

    np.save(log_dir+'mass_sig_Y.npy',np.array(mass_sig_Y))
    np.save(log_dir+'mass_bg_Y.npy',np.array(mass_bg_Y))#NOTE: ADDED 10/21/22 #NOTE: ADDED 12/1/22 Removed mask index here and above.

    # Plot fit
    x = np.linspace(low_high[0],low_high[1],bins)#mass_sig_Y[~mass_sig_Y.mask]
    y = hdata[0]
    plt.plot(x, func(x, *optParams), color='r')
    plt.plot(x, sig(x, *optParams[0:5]), color='tab:purple')
    plt.plot(x, bg(x, *optParams[5:]), color='b') #NOTE: CHANGED 10/21/22 optParams[5:] -> optParams[-1] 
    plt.hist(x, weights=y-bg(x, *optParams[5:]), bins=bins, range=low_high, histtype='step', alpha=0.5, color='b') #NOTE: CHANGED 10/21/22 optParams[5:] -> optParams[-1]

    #NOTE ADDED CHI2 CALCULATION 11/17/22
    r = np.divide(y - func(x, *optParams),np.sqrt([el if el>0 else 1 for el in func(x, *optParams)])) #NOTE: TAKE A LOOK AT https://root.cern.ch/doc/master/classRooChi2Var.html
    print("DEBUGGING: r = ",r)#DEBUGGING
    chi2 = np.sum(np.square(r))
    print("DEBUGGING: chi2 = ",chi2)#DEBUGGING
    ndf = len(y) - len(optParams)
    print("DEBUGGING: ndf = ",len(y)," - ",len(optParams)," = ",ndf)#DEBUGGING
    chi2ndf = chi2/ndf
    print("DEBUGGING: chi2/ndf = ",chi2ndf)#DEBUGGING
    
    # Setup legend entries for fit info
    lg = "Fit Info\n-------------------------\n"
    #lg += f"N = {round(optParams[0],0)}±{round(pcov[0,0],4)}\n"
    lg += f"α = {round(optParams[1],3)}±{round(pcov[1,1],7)}\n"
    lg += f"n = {round(optParams[2],3)}±{round(pcov[2,2],2)}\n"
    lg += f"μ = {round(optParams[3],5)}±{round(pcov[3,3],10)}\n"
    lg += f"σ = {round(optParams[4],5)}±{round(pcov[4,4],10)}\n"
    lg += f"$\chi^{2}$ = {round(chi2ndf,5)}\n"
    #lg += f"A = {round(optParams[5],0)}±{round(pcov[5,5],2)}\n"
    #lg += f"β = {round(optParams[6],0)}±{round(pcov[6,6],2)}\n"
    #lg += f"M = {round(optParams[7],2)}±{round(pcov[7,7],7)}\n"
    plt.text(low_high[1]-(low_high[1]-low_high[0])/3,0.5*max(hdata[0]),lg,fontsize=20,linespacing=1.25)
    
    # Show the graph
    plt.hist(mass_bg_Y[~mass_bg_Y.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'eval_metrics_mass_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot mass decisions separated into signal/background
    bins = 100
    # low_high = (1.1,1.13)
    f = plt.figure(figsize=figsize)
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
    f = plt.figure(figsize=figsize)
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
    f = plt.figure(figsize=figsize)
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
    f = plt.figure(figsize=figsize)
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
    f = plt.figure(figsize=figsize)
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
    f = plt.figure(figsize=figsize)
    plt.title('NN-identified bg mass distribution MC-matched')
    plt.hist(mass_bg_true[~mass_bg_true.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='true')
    plt.hist(mass_bg_false[~mass_bg_false.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='false')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'mc_matched_nn_bg_mass_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    print("DEBUGGING: np.unique(test_Y.detach().numpy()) = ",np.unique(test_Y.detach().numpy()))#DEBUGGING

    # Get ROC curve
    pfn_fp, pfn_tp, threshs = roc_curve(test_Y.detach().numpy(), probs_Y[:,1].detach().numpy())

    # Get area under the ROC curve

    auc = roc_auc_score(np.squeeze(test_Y.detach().numpy()), probs_Y[:,1].detach().numpy())
    if verbose: print(f'AUC = {auc:.4f}')

    # Create matplotlib plots for ROC curve and testing decisions
    f = plt.figure(figsize=(16,10))

    # Get some nicer plot settings 
    # plt.rcParams['figure.figsize'] = (4,4)#DEBUGGING
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
    plt.legend([model.name+f": AUC={auc:.4f} acc={test_acc:.4f}"],loc='lower left', frameon=False)
    f.savefig(os.path.join(log_dir,model.name+"_ROC_"+datetime.datetime.now().strftime("%F")+dataset+".png"))

    ##########################################################
    # Plot testing decisions
    bins = 100
    low = min(np.min(p) for p in probs_Y[:,1].detach().numpy())
    high = max(np.max(p) for p in probs_Y[:,0].detach().numpy())
    low_high = (low,high)
    f = plt.figure(figsize=figsize)
    plt.clf()
    #plt.hist(probs_Y[:,1].detach().numpy(), color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist1')
    #plt.hist(probs_Y[:,0].detach().numpy(), color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist2')
    plt.hist(probs_Y[:,0].detach().numpy(), color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', log=True, density=False, label='hist2') #NOTE: log=True added 11/17/22
    plt.xlabel('output')
    plt.ylabel('counts')
    plt.xlim(0.0,1.0) #NOTE: ADDED 11/17/22
    f.savefig(os.path.join(log_dir,model.name+"_test_decisions_"+datetime.datetime.now().strftime("%F")+dataset+".png"))

    #---------- ADDED ----------#

    

    ################################################################################
    # Plot testing decisions                                                                                                                                             
    bins = 100
    low = min(np.min(p) for p in probs_Y[:,1].detach().numpy())
    high = max(np.max(p) for p in probs_Y[:,0].detach().numpy())
    low_high = (low,high)
    f = plt.figure(figsize=figsize)
    plt.clf()
    #plt.hist(probs_Y[:,1].detach().numpy(), color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist1')
    #plt.hist(probs_Y[:,0].detach().numpy(), color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist2')
    separated_mass = [probs_Y[~mass_sig_MC.mask][:,0].detach().numpy(),probs_Y[~mass_bg_MC.mask][:,0].detach().numpy()]
    np.save(log_dir+'prediction.npy',prediction.detach().numpy())#NOTE: ADDED 12/29/22
    np.save(log_dir+'probs_Y.npy',probs_Y.detach().numpy())#NOTE: ADDED 12/29/22
    np.save(log_dir+'separated_probs_sig.npy',separated_mass[0])#NOTE: ADDED 12/29/22
    np.save(log_dir+'separated_probs_b.npy',separated_mass[1])#NOTE: ADDED 12/29/22
    plt.hist(separated_mass, range=low_high, bins=bins, alpha=0.5, histtype='stepfilled', log=True, stacked=True, density=False, label=['signal','background']) #NOTE: log=True added 11/17/22
    plt.xlabel('output')
    plt.ylabel('counts')
    plt.xlim(0.0,1.0) #NOTE: ADDED 11/17/22
    plt.legend()#NOTE: ADDED 12/29/22
    f.savefig(os.path.join(log_dir,model.name+"_STACKED_test_decisions_"+datetime.datetime.now().strftime("%F")+dataset+".png"))

    #################################################################################
    x_multi_sig = [mass_sig_false[~mass_sig_false.mask],mass_sig_true[~mass_sig_true.mask]]
    x_multi_bg  = [mass_bg_true[~mass_bg_true.mask], mass_bg_false[~mass_bg_false.mask]]
    
    bins = 100
    low_high = (1.08,1.24)#(1.1,1.13)
    f = plt.figure(figsize=figsize)
    plt.title('Invariant mass spectrum')
    hdata__ = plt.hist(x_multi_sig, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=('False Sigal','True Signal')) #NOTE: MAKE SURE THESE MATCH UP!!!

    # Plot fit
    x = np.linspace(low_high[0],low_high[1],bins)#mass_sig_Y[~mass_sig_Y.mask]
    y = hdata[0]
    plt.plot(x, func(x, *optParams), color='r')
    plt.plot(x, sig(x, *optParams[0:5]), color='tab:purple')
    plt.plot(x, bg(x, *optParams[5:]), color='b') #NOTE: CHANGED 10/21/22 optParams[5:] -> optParams[-1] 
    #plt.hist(x, weights=y-bg(x, *optParams[5:]), bins=bins, range=low_high, histtype='step', alpha=0.5, color='b') #NOTE: CHANGED 10/21/22 optParams[5:] -> optParams[-1] #NOTE: DEBUGGING COMMENTED OUT 11/4/22

    #r = y - func(x, *optParams)
    r = np.divide(y - func(x, *optParams),np.sqrt([el if el>0 else 1 for el in func(x, *optParams)])) #NOTE: TAKE A LOOK AT https://root.cern.ch/doc/master/classRooChi2Var.html
    print("DEBUGGING: r = ",r)#DEBUGGING
    chi2 = np.sum(np.square(r))
    print("DEBUGGING: chi2 = ",chi2)#DEBUGGING
    ndf = len(y) - len(optParams)
    print("DEBUGGING: ndf = ",len(y)," - ",len(optParams)," = ",ndf)#DEBUGGING
    chi2ndf = chi2/ndf
    print("DEBUGGING: chi2/ndf = ",chi2ndf)#DEBUGGING

    # Setup legend entries for fit info
    lg = "Fit Info\n-------------------------\n"
    #lg += f"N = {round(optParams[0],0)}±{round(pcov[0,0],4)}\n"
    lg += f"α  = {round(optParams[1],3)}±{round(pcov[1,1],7)}\n"
    lg += f"n  = {round(optParams[2],3)}±{round(pcov[2,2],2)}\n"
    lg += f"μ  = {round(optParams[3],5)}±{round(pcov[3,3],10)}\n"
    lg += f"σ  = {round(optParams[4],5)}±{round(pcov[4,4],10)}\n"
    lg += f"$\chi^{2}$ = {round(chi2ndf,5)}\n"
    #lg += f"A = {round(optParams[5],0)}±{round(pcov[5,5],2)}\n"
    #lg += f"β = {round(optParams[6],0)}±{round(pcov[6,6],2)}\n"
    #lg += f"M = {round(optParams[7],2)}±{round(pcov[7,7],7)}\n"
    plt.text(low_high[1]-(low_high[1]-low_high[0])/3,1/2*max(hdata[0]),lg,fontsize=20,linespacing=1.25) #NOTE: MAKE THESE PARAMS OPTIONS

    # Show the graph
    # plt.hist(mass_bg_Y[~mass_bg_Y.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background') #NOTE: COMMENTED OUT FOR DEBUGGING
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('$M_{p\pi^{-}}$ (GeV)')

    print("DEBUGGING: MADE IT! printing out graph to ",os.path.join(log_dir,'parent_decomposition_full_with_fit.png'))#DEBUGGING ADDED 11/4/22
    f.savefig(os.path.join(log_dir,'parent_decomposition_full_with_fit.png'))

    ################################################################################

    #-------- END ADDED --------#

    return (auc,test_acc) #NOTE: Needed for optimization_study() below.


def optimization_study(
    args,
    device=torch.device('cpu'),
    log_interval=10,
    log_dir="logs/",
    save_path="model",
    verbose=True):

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

        # Setup data and model #NOTE: DO THIS HERE SINCE IT DEPENDS ON BATCH SIZE. #TODO: NOTE HOPEFULLY ALL BELOW WORKS...
        train_dataloader, val_dataloader, eval_loader, nclasses, nfeatures, nfeatures_edge = [None for i in range(6)]
        if len(args.indices)>3:
            train_dataloader, val_dataloader, eval_loader, nclasses, nfeatures, nfeatures_edge = load_graph_dataset(dataset=args.dataset, prefix=args.prefix, 
                                                    split=args.split, max_events=args.max_events, indices=args.indices,
                                                    num_workers=args.nworkers, batch_size=batch_size) 
        elif len(args.indices)==3:
            train_dataloader, val_dataloader, nclasses, nfeatures, nfeatures_edge = load_graph_dataset(dataset=args.dataset, prefix=args.prefix, 
                                                    split=args.split, max_events=args.max_events, indices=args.indices,
                                                    num_workers=args.nworkers, batch_size=batch_size)
        else:
            train_dataloader, val_dataloader, nclasses, nfeatures, nfeatures_edge = load_graph_dataset(dataset=args.dataset, prefix=args.prefix, 
                                                    split=args.split, max_events=args.max_events,
                                                    num_workers=args.nworkers, batch_size=batch_size)

        # Instantiate model, optimizer, scheduler, and loss
        model = GIN(nlayers,nmlp,nfeatures,hdim,nclasses,do,args.learn_eps,args.npooling,args.gpooling).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=args.patience,
            threshold=args.thresh, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=args.verbose)
        if step==0:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=args.verbose)
        if step>0:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma, verbose=args.verbose)
        criterion = nn.CrossEntropyLoss()

        # Make sure log/save directories exist
        trialdir = 'trial_'+datetime.datetime.now().strftime("%F")+'_'+args.dataset+'_'+args.study_name+'_'+str(trial.number+1)+'/' #NOTE +'/' ADDED 11/17/22
        try:
            os.makedirs(args.log+'/'+trialdir) #NOTE: Do NOT use os.path.join() here since it requires that the directory already exist.
        except FileExistsError:
            if args.verbose: print("Directory exists: ",os.path.join(args.log,trialdir))
        trialdir = os.path.join(args.log,trialdir)

        # Show model if requested
        if args.verbose: print(model)

        # Logs for matplotlib plots
        logs = train(
                    args,
                    model,
                    device,
                    train_dataloader,
                    val_dataloader,
                    optimizer,
                    scheduler,
                    criterion,
                    args.epochs,
                    dataset=args.dataset,
                    prefix=args.prefix,
                    log_dir=trialdir,
                    verbose=args.verbose
                    )

        # Get testing AUC
        metrics = evaluate(
            model,
            device,
            eval_loader=eval_loader, #TODO: IMPLEMENT THIS
            dataset=args.dataset, 
            prefix=args.prefix,
            split=args.split,
            max_events=args.max_events,
            log_dir=trialdir,
            verbose=True
        )

        return 1.0 - metrics[0] #NOTE: This is so you maximize AUC since can't figure out how to create sqlite3 study with maximization at the moment 8/5/22

    #----- MAIN PART -----#
    
    # Load or create pruner, sampler, and study
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    sampler = TPESampler() #TODO: Add command line option for selecting different sampler types.
    study = optuna.create_study(storage='sqlite:///'+args.db_path, sampler=sampler,pruner=pruner, study_name=args.study_name, direction="minimize", load_if_exists=True) #TODO: Add options for different SQL programs: Postgre, MySQL, etc.

    # Run optimization
    study.optimize(objective, n_trials=args.ntrials, timeout=args.timeout, gc_after_trial=True) #NOTE: gc_after_trial=True is to avoid OOM errors see https://optuna.readthedocs.io/en/stable/faq.html#out-of-memory-gc-collect
    trial = study.best_trial

    if verbose:
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

def optimization_study_dagnn(
    args,device=torch.device('cpu'),
    log_interval=10,
    log_dir="logs/",
    save_path="torch_models",
    verbose=True):
    #NOTE: As of right now log_dir='logs/' should end with the slash

    # # Load validation data
    # test_dataset = GraphDataset(args.prefix+args.dataset)
    # test_dataset.load()
    # test_dataset = Subset(test_dataset,range(int(len(test_dataset)*args.split),len(test_dataset))) #NOTE: This is currently same as validation data...

    # def setup(rank, world_size,method="gloo"):
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '12355'

    #     # initialize the process group
    #     dist.init_process_group(method, rank=rank, world_size=world_size)

    # def cleanup():
    #     dist.destroy_process_group()
    
    # def fn_to_dist(rank, world_size):

    #     print(f"Running basic DDP example on rank {rank}.")
    #     setup(rank, world_size)

    #     fn_to_dist(**kwargs)#NOTE: Replace with train or train_dagnn

    #     cleanup()


    # def run_dist(fn_to_dist, world_size, **kwargs): #NOTE: wrap train or train_dagnn in run_dist
    #     mp.spawn(fn_to_dist,
    #             args=(world_size, **kwargs,),
    #             nprocs=world_size,
    #             join=True)

    def objective(trial):

        # Get parameter suggestions for trial
        #TODO: Add suggestions for all other hyperparameters DONE
        #TODO: Add new options to args in test_dagnn.py DONE
        #TODO: Update model creation below... DONE
        #TODO: Update BCELoss->CrossEntropyLoss below DONE
        alpha = args.alpha[0] if args.alpha[0] == args.alpha[1] else trial.suggest_float("alpha",args.alpha[0],args.alpha[1]) #NOTE: 11/17/22 CHANGED suggest_int->suggest_float
        batch_size = args.batch[0] if args.batch[0] == args.batch[1] else trial.suggest_int("batch_size",args.batch[0],args.batch[1]) 
        nlayers = args.nlayers[0] if args.nlayers[0] == args.nlayers[1] else trial.suggest_int("nlayers",args.nlayers[0],args.nlayers[1])
        nmlp  = args.nmlp[0] if args.nmlp[0] == args.nmlp[1] else trial.suggest_int("nmlp",args.nmlp[0],args.nmlp[1])
        hdim  = args.hdim[0] if args.hdim[0] == args.hdim[1] else trial.suggest_int("hdim",args.hdim[0],args.hdim[1])
        nmlp_head  = args.nmlp_head[0] if args.nmlp_head[0] == args.nmlp_head[1] else trial.suggest_int("nmlp_head",args.nmlp_head[0],args.nmlp_head[1])
        hdim_head  = args.hdim_head[0] if args.hdim_head[0] == args.hdim_head[1] else trial.suggest_int("hdim_head",args.hdim_head[0],args.hdim_head[1])
        do    = args.dropout[0] if args.dropout[0] == args.dropout[1] else trial.suggest_float("do",args.dropout[0],args.dropout[1])
        lr    = args.lr[0] if args.lr[0] == args.lr[1] else trial.suggest_float("lr",args.lr[0],args.lr[1],log=True)
        lr_c  = args.lr_c[0] if args.lr_c[0] == args.lr_c[1] else trial.suggest_float("lr_c",args.lr_c[0],args.lr_c[1],log=True)
        lr_d  = args.lr_d[0] if args.lr_d[0] == args.lr_d[1] else trial.suggest_float("lr_d",args.lr_d[0],args.lr_d[1],log=True)
        step  = args.step[0] if args.step[0] == args.step[1] else trial.suggest_int("step",args.step[0],args.step[1])
        gamma = args.gamma[0] if args.gamma[0] == args.gamma[1] else trial.suggest_float("gamma",args.gamma[0],args.gamma[1])
        max_epochs = args.epochs

        # Setup data and model #NOTE: DO THIS HERE SINCE IT DEPENDS ON BATCH SIZE. #TODO: NOTE HOPEFULLY ALL BELOW WORKS...
        train_dataloader, val_dataloader, eval_loader, dom_train_loader, dom_val_loader, nclasses, nfeatures, nfeatures_edge = [None for i in range(8)]
        if len(args.indices)>3:
            train_loader, val_loader, eval_loader, nclasses, nfeatures_node, nfeatures_edge = load_graph_dataset(dataset=args.dataset, prefix=args.prefix, 
                                                    split=args.split, max_events=args.max_events, indices=args.indices,
                                                    num_workers=args.nworkers, batch_size=batch_size)

            dom_train_loader, dom_val_loader, dom_nclasses, dom_nfeatures_node, dom_nfeatures_edge = load_graph_dataset(dataset=args.dom_dataset, prefix=args.dom_prefix, 
                                                    split=args.split, max_events=args.max_events, indices=args.indices[0:3],
                                                    num_workers=args.nworkers, batch_size=batch_size) 
        elif len(args.indices)==3:
            train_loader, val_loader, eval_loader, nclasses, nfeatures_node, nfeatures_edge = load_graph_dataset(dataset=args.dataset, prefix=args.prefix, 
                                                    split=args.split, max_events=args.max_events, indices=args.indices,
                                                    num_workers=args.nworkers, batch_size=batch_size)

            dom_train_loader, dom_val_loader, dom_nclasses, dom_nfeatures_node, dom_nfeatures_edge = load_graph_dataset(dataset=args.dom_dataset, prefix=args.dom_prefix, 
                                                    split=args.split, max_events=args.max_events, indices=args.indices,
                                                    num_workers=args.nworkers, batch_size=batch_size)
        else:
            train_loader, val_loader, nclasses, nfeatures_node, nfeatures_edge = load_graph_dataset(dataset=args.dataset, prefix=args.prefix, 
                                                    split=args.split, max_events=args.max_events,
                                                    num_workers=args.nworkers, batch_size=batch_size)

            dom_train_loader, dom_val_loader, dom_nclasses, dom_nfeatures_node, dom_nfeatures_edge = load_graph_dataset(dataset=args.dom_dataset, prefix=args.dom_prefix, 
                                                    split=args.split, max_events=args.max_events,
                                                    num_workers=args.nworkers, batch_size=batch_size)

        # Check that # classes and data dimensionality at nodes and edges match between training and domain data
        if nclasses!=dom_nclasses or nfeatures_node!=dom_nfeatures_node or nfeatures_edge!=dom_nfeatures_edge:
            print("*** ERROR *** mismatch between graph structure for domain and training data!")
            print("EXITING...")
            return

        n_domains = 2
        nfeatures = nfeatures_node

        # Create models
        model = GIN(nlayers, nmlp, nfeatures,
                hdim, hdim, do, args.learn_eps, args.npooling,
                args.gpooling).to(device)
        #NOTE: OLD BELOW 7/22/22
        # classifier = Classifier(input_size=hdim,num_classes=nclasses).to(device)
        # discriminator = Discriminator(input_size=hdim,num_classes=n_domains-1).to(device)
        classifier = MLP_SIGMOID(nmlp_head, hdim, hdim_head, nclasses).to(device)
        discriminator = MLP_SIGMOID(nmlp_head, hdim, hdim_head, n_domains).to(device)

        # # Make models parallel if multiple gpus available
        # if device.type=='cuda' and device.index==None:
        #     model = DataParallel(model)
        #     classifier = DataParallel(classifier)
        #     discriminator = DataParallel(discriminator)

        # Create optimizers
        model_optimizer = optim.Adam(model.parameters(), lr=lr)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr_c)
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr_d)

        # Create schedulers
        model_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, mode='min', factor=gamma, patience=args.patience,
            threshold=args.thresh, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=args.verbose)
        if step==0:
            model_scheduler = optim.lr_scheduler.ExponentialLR(model_optimizer, gamma, last_epoch=-1, verbose=args.verbose)
        if step>0:
            model_scheduler = optim.lr_scheduler.StepLR(model_optimizer, step_size=step, gamma=gamma, verbose=args.verbose)

        # Create loss functions
        train_criterion = nn.CrossEntropyLoss()
        dom_criterion   = nn.CrossEntropyLoss()

        # Make sure log/save directories exist
        trialdir = 'trial_'+datetime.datetime.now().strftime("%F")+'_'+args.dataset+'_'+args.study_name+'_'+str(trial.number+1)+'/' #NOTE +'/' ADDED 11/17/22 
        try:
            os.makedirs(args.log+'/'+trialdir) #NOTE: Do NOT use os.path.join() here since it requires that the directory already exist.
        except FileExistsError:
            if args.verbose: print("Directory exists: ",os.path.join(args.log,trialdir))
        trialdir = os.path.join(args.log,trialdir)

        # Show model if requested
        if args.verbose: print(model)

        # Logs for matplotlib plots
        logs = train_dagnn(
                            args,
                            model,
                            classifier,
                            discriminator,
                            device,
                            train_loader,
                            val_loader,
                            dom_train_loader,
                            dom_val_loader,
                            model_optimizer,
                            classifier_optimizer,
                            discriminator_optimizer,
                            model_scheduler,
                            train_criterion,
                            dom_criterion,
                            alpha,#TODO: Commented out for DEBUGGING
                            args.epochs,
                            dataset=args.dataset,
                            prefix=args.prefix,
                            log_interval=args.log_interval,
                            log_dir=trialdir,
                            save_path=args.save_path,
                            verbose=args.verbose
                        )


        # Setup data and model
        nclasses, nfeatures, nfeatures_edge = get_graph_dataset_info(dataset=args.dataset, prefix=args.prefix)

        # _model = GIN(nlayers, nmlp, nfeatures,
        #         hdim, hdim, do, args.learn_eps, args.npooling,
        #         args.gpooling).to(device)
        # _classifier = Classifier(input_size=hdim,num_classes=nclasses).to(device)
        # print("INFO: LOADING: ",os.path.join(trialdir,args.name+'_model_weights'))#DEBUGGING
        # print("INFO: LOADING: ",os.path.join(trialdir,args.name+'_classifier_weights'))#DEBUGGING
        # _model.load_state_dict(torch.load(os.path.join(trialdir,args.name+'_model_weights'),map_location=args.device))
        # _classifier.load_state_dict(torch.load(os.path.join(trialdir,args.name+'_classifier_weights'),map_location=args.device))

        model_concatenate = Concatenate([ model, classifier])
        model_concatenate.eval()

        # Get testing AUC
        metrics = evaluate(
            model_concatenate,
            device,
            dataset=args.dataset,
            prefix=args.prefix,
            eval_loader=eval_loader,
            split=args.split,
            max_events=args.max_events,
            log_dir=trialdir,
            verbose=True
        )

        return 1.0 - metrics[0] #NOTE: This is so you maximize AUC since can't figure out how to create sqlite3 study with maximization at the moment 8/9/22

    #----- MAIN PART -----#
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    sampler = TPESampler() #TODO: Add command line option for selecting different sampler types.
    study = optuna.create_study(storage='sqlite:///'+args.db_path, sampler=sampler,pruner=pruner, study_name=args.study_name, direction="minimize", load_if_exists=True) #TODO: Add options for different SQL programs: Postgre, MySQL, etc.
    study.optimize(objective, n_trials=args.ntrials, timeout=args.timeout, gc_after_trial=True) #NOTE: gc_after_trial=True is to avoid OOM errors see https://optuna.readthedocs.io/en/stable/faq.html#out-of-memory-gc-collect
    trial = study.best_trial

    if verbose:
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

def evaluate_on_data(model,device,dataset="", prefix="", split=1.0, log_dir="logs/",verbose=True,roc_cut=None,model1=None,use_umap=True):

    #TODO: Make these options...
    plt.rc('font', size=20) #controls default text size                                                                                                                     
    plt.rc('axes', titlesize=25) #fontsize of the title                                                                                                                     
    plt.rc('axes', labelsize=25) #fontsize of the x and y labels                                                                                                            
    plt.rc('xtick', labelsize=20) #fontsize of the x tick labels                                                                                                            
    plt.rc('ytick', labelsize=20) #fontsize of the y tick labels                                                                                                            
    plt.rc('legend', fontsize=20) #fontsize of the legend

    figsize=(16,10) #NOTE: ADDED 9/30/22

    # Load validation data
    test_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    test_dataset.load()
    test_dataset = Subset(test_dataset,range(int(len(test_dataset)*split)))

    model.eval()
    model      = model.to(device)
    test_bg    = dgl.batch(test_dataset.dataset.graphs[test_dataset.indices.start:test_dataset.indices.stop])#TODO: Figure out nicer way to use subset
    test_bg    = test_bg.to(device)
    prediction = model(test_bg)
    #NOTE: DEBUGGING: 12/29/22
    is_dagin = False
    try:
        nmodels = len(model.models)
        is_dagin = nmodels==2
    except AttributeError as ae:
        print("FROM utils.py:evaluate_on_data() ERROR: could not access attribute model.models")
        print(ae)
    #NOTE: END DEBUGGING: 12/29/22
    probs_Y    = torch.softmax(prediction, 1) if not is_dagin else prediction
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1) if roc_cut is None else torch.tensor([1 if el>roc_cut else 0 for el in probs_Y[:,1]],dtype=torch.long) #NOTE: ADDED 10/25/22

    #NOTE: ADDED 11/10/22
    if model1 is not None:
        model1.eval()
        model1      = model1.to(device)
        test_bg1    = dgl.batch(test_dataset.dataset.graphs[test_dataset.indices.start:test_dataset.indices.stop])#TODO: Figure out nicer way to use subset
        test_bg1    = test_bg1.to(device)
        prediction1 = model1(test_bg1)
        probs_Y1    = torch.softmax(prediction1, 1)
        argmax_Y1   = torch.max(probs_Y1, 1)[1].view(-1, 1) if roc_cut is None else torch.tensor([1 if el>roc_cut else 0 for el in probs_Y[:,1]],dtype=torch.long) #NOTE: ADDED 10/25/22
        argmax_Y    = torch.min(argmax_Y,argmax_Y1) if roc_cut is None else torch.tensor([1 if el>roc_cut and probs_Y1[:,1][idx]>roc_cut else 0 for idx, el in enumerate(probs_Y[:,1])],dtype=torch.long)

    # Copy arrays back to CPU
    probs_Y  = probs_Y.cpu()
    argmax_Y = argmax_Y.cpu()

    print("DEBUGGING: test_dataset.indices.start = ",test_dataset.indices.start)#DEBUGGING
    print("DEBUGGING: test_dataset.indices.stop = ",test_dataset.indices.stop)#DEBUGGING
    print("DEBUGGING: np.min(ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float()])) = ",np.min( ma.array( test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float() )))#DEBUGGING

    # Get separated mass distributions
    mass_sig_Y    = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),mask=~(argmax_Y == 1))
    mass_bg_Y     = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),mask=~(argmax_Y == 0))

    

    #DEBUGGING #NOTE JUST FOR EVALUATING NO GNN FIT WITH EXACT STATISTICS TO HAVE COMPARABLE FOM
    #mass_sig_Y    = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),mask=False)
    #DEBUGGING

    #try: #DEBUGGING #ADDED 11/29/22
    import umap
    import seaborn as sns
    from sklearn.manifold import TSNE
    reducer = umap.UMAP() if use_umap else TSNE(2)
    extractor = None
    latent_repr = None
    model_type = None
    try:
        model_type='DAGIN'
        extractor = model.models[0]
        latent_repr = extractor.get_latent_repr(test_bg).detach()
        print("DEBUGGING: type(latent_repr) = ",type(latent_repr[0]))#DEBUGGING
        print("DEBUGGING: latent_repr[0] = ",latent_repr[0])#DEBUGGING
        print("DEBUGGING: len(latent_repr) = ",len(latent_repr))#DEBUGGING
    except AttributeError as ae:
        model_type='GIN'
        print("AttributeError: could not access model.models")#DEBUGGING
        print(ae)
        extractor = model
        latent_repr = extractor.get_latent_repr(test_bg).detach()
        print("DEBUGGING: type(latent_repr) = ",type(latent_repr[0]))#DEBUGGING
        print("DEBUGGING: latent_repr[0] = ",latent_repr[0])#DEBUGGING
        print("DEBUGGING: len(latent_repr) = ",len(latent_repr))#DEBUGGING

    print("DEBUGGING: type(latent_repr) = ",type(latent_repr))#DEBUGGING
    print("DEBUGGING: latent_repr = ",latent_repr)#DEBUGGING
    embedding = reducer.fit_transform(latent_repr)
    print("DEBUGGING: type(embedding) = ",type(embedding))#DEBUGGING
    print("DEBUGGING: embedding.shape = ",embedding.shape)#DEBUGGING

    visualization_method = 'umap' if use_umap else 'tsne'

    np.save(log_dir+'latent_repr.npy',np.array(latent_repr))#DEBUGGING: NOTE ADDED 11/30/22
    np.save(log_dir+visualization_method+'_embedding.npy',np.array(embedding))#DEBUGGING: NOTE ADDED 11/30/22

    f = plt.figure(figsize=figsize)
    plt.scatter(embedding[:,0],embedding[:,1])
    plt.title(visualization_method+' projection of the '+model_type+' latent space presentation')
    f.savefig(os.path.join(log_dir,visualization_method+'_basic.png'))
    # Plot separated into true/false sig/bg

    em_sig = ma.array(embedding,mask=[[el,el] for el in mass_sig_Y.mask])
    em_bg = ma.array(embedding,mask=[[el,el] for el in mass_bg_Y.mask])

    #visualization_method = 'umap' if use_umap else 'tsne'

    # Set plotting options
    alpha = 0.05
    mew = 0.0

    # Plot visualization method decisions separated into signal/background
    f, ax = plt.subplots(figsize=figsize)
    plt.title('Separated '+visualization_method+' distribution')
    ax.scatter(em_sig[:,0][~em_sig.mask[:,0]],em_sig[:,1][~em_sig.mask[:,1]], alpha=alpha, linewidths=mew, label='sig')
    ax.scatter(em_bg[:,0][~em_bg.mask[:,0]],em_bg[:,1][~em_bg.mask[:,1]], alpha=alpha, linewidths=mew, label='bg')
    ax.legend(loc='upper left', frameon=False)
    f.savefig(os.path.join(log_dir,visualization_method+'_nn_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot visualization method decisions separated into signal/background
    f, ax = plt.subplots(figsize=figsize)
    plt.title(visualization_method+' sig distribution')
    ax.scatter(em_sig[:,0][~em_sig.mask[:,0]],em_sig[:,1][~em_sig.mask[:,1]], color="tab:blue", alpha=alpha, linewidths=mew, label='sig')
    #ax.scatter(em_bg[:,0][~em_bg.mask[:,0]],em_bg[:,1][~em_bg.mask[:,1]], alpha=alpha, linewidths=mew, label='bg')
    ax.legend(loc='upper left', frameon=False)
    f.savefig(os.path.join(log_dir,visualization_method+'_nn_sig_only_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot visualization decisions separated into signal/background
    f, ax = plt.subplots(figsize=figsize)
    plt.title(visualization_method+' bg distribution')
    #ax.scatter(em_sig[:,0][~em_sig.mask[:,0]],em_sig[:,1][~em_sig.mask[:,1]], alpha=alpha, linewidths=mew, label='sig')
    ax.scatter(em_bg[:,0][~em_bg.mask[:,0]],em_bg[:,1][~em_bg.mask[:,1]], color="tab:orange", alpha=alpha, linewidths=mew, label='bg')
    ax.legend(loc='upper left', frameon=False)
    f.savefig(os.path.join(log_dir,visualization_method+'_nn_bg_only_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    #except Exception:
    #    print("COULD NOT IMPORT UMAP OR ACCESS MODEL EXTRACTOR ELEMENT...")#DEBUGGING ADDED 11/29/22

    ################################################### #NOTE: COMMENTED OUT 10/21/22
    ## Define fit function
    #def func(x, N, beta, m, loc, scale, A, B, C):
    #    return N*crystalball.pdf(-x, beta, m, -loc, scale) + A*(1 - B*(x - C)**2)
    #    
    #def sig(x, N, beta, m, loc, scale):
    #    return N*crystalball.pdf(-x, beta, m, -loc, scale)
    #    
    #def bg(x, A, B, C):
    #    return A*(1 - B*(x - C)**2)
    ###################################################

    # Plot mass decisions separated into signal/background
    bins = 100
    low_high = (1.08,1.24)#(1.1,1.13)
    f = plt.figure(figsize=(16,10))
    plt.title('Separated mass distribution')
    # mass_all = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float())
    # hdata = plt.hist(mass_all, color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    hdata = plt.hist(mass_sig_Y[~mass_sig_Y.mask], color='tab:orange', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal') #NOTE: color='m' removed 9/28/22 #NOTE: DEBUGGING color='m', removed 11/4/22
    # plt.show()#DEBUGGING
    # return #DEBUGGING
    
    # Fit output of NN
    print("DEBUGGING: hdata[0][-1] = ",hdata[0][-1])#DEBUGGING
    print("DEBUGGING: hdata[0] = ",hdata[0])#DEBUGGING NOTE: ADDED 9/19/22
    N, beta, m, loc, scale, A, B, C = 10, 1, 1.112, 1.115, 0.008, np.max(hdata[0][-10:-1]), 37, 1.24 #NOTE: Updated with np.max 10/28/22
    if A==0: A = 0.1#DEBUGGING
    d_N, d_beta, d_m, d_loc, d_scale, d_A, d_B, d_C = N/0.01, beta/0.1, m/0.1, loc/0.1, scale/0.01, A/10, B/0.1, C/1
    #d_N, d_beta, d_m, d_loc, d_scale, d_A, d_B, d_C = N/0.001, beta/0.001, m/0.01, loc/0.001, scale/0.001, A/1, B/0.1, C/5
    parsMin = [N-d_N, beta-d_beta, m-d_m, loc-d_loc, scale-d_scale, B-d_B]# [N-d_N, beta-d_beta, m-d_m, loc-d_loc, scale-d_scale, A-d_A, B-d_B, C-d_C] #NOTE: CHANGED 10/21/22
    parsMax = [N+d_N, beta+d_beta, m+d_m, loc+d_loc, scale+d_scale, B+d_B]# [N+d_N, beta+d_beta, m+d_m, loc+d_loc, scale+d_scale, A+d_A, B+d_B, C+d_C] #NOTE: CHANGED 10/21/22
    print(parsMin)#DEBUGGING
    print(parsMax)#DEBUGGING
    #N/=8#NOTE: DEBUGGING ADDED 9/19/22

    ################################################# #NOTE: UPDATED FUNCTION DEFINITIONS 10/21/22
    # Define fit function
    def func(x, N, beta, m, loc, scale, B, A=A, C=C):
        return N*crystalball.pdf(-x, beta, m, -loc, scale) + A*(1 - B*(x - C)**2)

    def sig(x, N, beta, m, loc, scale):
        return N*crystalball.pdf(-x, beta, m, -loc, scale)

    def bg(x, B, A=A, C=C):
        return A*(1 - B*(x - C)**2)
    ##################################################

    optParams, pcov = opt.curve_fit(func, hdata[1][:-1], hdata[0], method='trf', bounds=(parsMin,parsMax))
    #optParams = [N, beta, m, loc, scale, A, B, C] #DEBUGGING
    #pcov = np.array([[0.0 for i in optParams] for i in optParams])#DEBUGGING

    np.save(log_dir+'mass_sig_Y.npy',np.array(mass_sig_Y))
    np.save(log_dir+'mass_bg_Y.npy',np.array(mass_bg_Y))#NOTE: ADDED 10/21/22 #NOTE: ADDED 12/1/22 Removed mask index here and above.

    np.save(log_dir+'mass_sig_Y_mask.npy',mass_sig_Y.mask)#NOTE: ADDED 12/1/22
    np.save(log_dir+'mass_bg_Y_mask.npy',mass_bg_Y.mask)#NOTE: ADDED 12/1/22
    
    # Plot fit
    x = np.linspace(low_high[0],low_high[1],bins)#mass_sig_Y[~mass_sig_Y.mask]
    y = hdata[0]
    
    plt.plot(x, func(x, *optParams), color='r')
    plt.plot(x, sig(x, *optParams[0:5]), color='tab:purple')
    plt.plot(x, bg(x, *optParams[5:]), color='b') #NOTE: CHANGED 10/21/22 optParams[5:] -> optParams[-1]
    bghist = plt.hist(x, weights=y-bg(x, *optParams[5:]), bins=bins, range=low_high, histtype='step', alpha=0.5, color='b') #NOTE: CHANGED 10/21/22 optParams[5:] -> optParams[-1]

    r = np.divide(y - func(x, *optParams),np.sqrt([el if el>0 else 1 for el in func(x, *optParams)]))
    print("DEBUGGING: r = ",r)#DEBUGGING
    print("DEBUGGING: func(x, *optParams) = ",func(x, *optParams))#DEBUGGING
    chi2 = np.sum(np.square(r))
    print("DEBUGGING: chi2 = ",chi2)#DEBUGGING
    chi2ndf = chi2/len(optParams)
    print("DEBUGGING: chi2/ndf = ",chi2ndf)#DEBUGGING
    
    # Get S and N before and after? #DEBUGGING: ADDED
    import scipy.integrate as integrate
    mu      = optParams[3]
    sigma   = optParams[4]
    mmin    = mu - 2*sigma
    mmax    = mu + 2*sigma

    print("mmin = ",mmin)#DEBUGGING
    print("mmax = ",mmax)#DEBUGGING

    binwidth = (low_high[1]-low_high[0])/bins#KEEP!!!
    print("binwidth = ",binwidth)#DEBUGGING

    bin1 = int((mmin-low_high[0])/binwidth)
    bin2 = int((mmax-low_high[0])/binwidth)
    print("bin1 = ",bin1)#DEBUGGING
    print("bin2 = ",bin2)#DEBUGGING

    integral_bghist = sum(bghist[0][bin1:bin2])
    print("integral_bghist = ",integral_bghist)#DEBUGGING
    
    integral_tothist = sum(hdata[0][bin1:bin2])
    print("integral_tothist = ",integral_tothist)#DEBUGGING
    #if integral_tothist<=0:
    #    integral_tothist=1
    #    print("DEBUGGING: ERROR: integral_tothist=0 reassign to 1")#DEBUGGING
    fom = integral_bghist/np.sqrt(integral_tothist)
    print("FOM = ",(integral_bghist)/np.sqrt(integral_tothist))#DEBUGGING
    if integral_tothist<=0:
        integral_tothist=1
        print("DEBUGGING: ERROR: integral_tothist=0 reassign to 1")#DEBUGGING  
    print("S/N = ",(integral_bghist)/(integral_tothist))#DEBUGGING

    print("optParams = ",optParams)#DEBUGGING

    resultN = integrate.quad(lambda x: func(x, *optParams),mmin,mmax)[0] / binwidth
    resultS = integrate.quad(lambda x: sig(x, *optParams[0:5]),mmin,mmax)[0] / binwidth
    resultB = integrate.quad(lambda x: bg(x, *optParams[5:]),mmin,mmax)[0] / binwidth #NOTE: CHANGED 10/21/22 optParams[5:] -> optParams[-1]  

    print("resultN = ",resultN)#DEBUGGING
    print("resultS = ",resultS)#DEBUGGING
    print("resultB = ",resultB)#DEBUGGING

    # result_N = integrate.quad(func,mmin,mmax,args=optParams)
    # result_S = integrate.quad(sig,mmin,mmax,args=optParams[0:5])
    # result_B = integrate.quad(bg,mmin,mmax,args=optParams[5:])

    # print("result_N = ",result_N)#DEBUGGING
    # print("result_S = ",result_S)#DEBUGGING
    # print("result_B = ",result_B)#DEBUGGING
    
    # Setup legend entries for fit info
    lg = "Fit Info\n-------------------------\n"
    #lg += f"N = {round(optParams[0],0)}±{round(pcov[0,0],5)}\n" #NOTE: COMMENTED OUT FOR DEBUGGING
    lg += f"α  = {round(optParams[1],3)}±{round(pcov[1,1],5)}\n"
    lg += f"n  = {round(optParams[2],3)}±{round(pcov[2,2],5)}\n"
    lg += f"μ  = {round(optParams[3],5)}±{round(pcov[3,3],5)}\n"
    lg += f"σ  = {round(optParams[4],5)}±{round(pcov[4,4],5)}\n"
    lg += f"$\chi^{2}$ = {round(chi2ndf,5)}\n"
    #lg += f"A = {round(optParams[5],0)}±{round(pcov[5,6],5)}\n"
    #lg += f"β = {round(optParams[6],0)}±{round(pcov[6,6],5)}\n"
    #lg += f"M = {round(optParams[7],2)}±{round(pcov[7,7],5)}\n"
    plt.text(low_high[1]-(low_high[1]-low_high[0])/3,0.5*max(max(mass_bg_Y[~mass_bg_Y.mask] if len(mass_bg_Y[~mass_bg_Y.mask])>0 else [0]),max(hdata[0])),lg,fontsize=20,linespacing=1.25)
    
    # Show the graph
    #plt.hist(mass_bg_Y[~mass_bg_Y.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background') #NOTE: COMMENTED OUT FOR DEBUGGING
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
    f = plt.figure(figsize=figsize)
    plt.clf()
    #plt.hist(probs_Y[:,1].detach().numpy(), color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist1')
    #plt.hist(probs_Y[:,0].detach().numpy(), color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist2')
    plt.hist(probs_Y[:,0].detach().numpy(), color='b', range=low_high, bins=bins, histtype='stepfilled', log=True, alpha=0.5, density=False, label='hist2') #NOTE: #DEBUGGING: log=True added 11/9/22
    plt.xlabel('NN Output')
    plt.ylabel('Counts')
    #plt.ylim((0.0,4000))#NOTE: DEBUGGING ADDED 9/30/22
    f.savefig(os.path.join(log_dir,model.name+"_eval_decisions_"+datetime.datetime.now().strftime("%F")+dataset+".png"))

    np.save(log_dir+'prediction.npy',prediction.detach().numpy())#NOTE: ADDED 12/29/22
    np.save(log_dir+'probs_Y.npy',probs_Y.detach().numpy())#NOTE: ADDED 12/29/22

    return fom, resultN, resultS, resultB

#------------------------- Classes -------------------------#

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
        --------

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
        if self.inGraphs is not None and self.inLabels is not None:
            self.graphs, self.labels = self.inGraphs, self.inLabels #DEBUGGING: COMMENTED OUT: torch.LongTensor(self.inLabels)
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

    def shuffle(self):
        """
        Randomly shuffle dataset graphs and labels.
        """
        indices = np.array([i for i in range(len(self.graphs))])
        np.random.shuffle(indices) #NOTE: In-place method
        self.labels = torch.stack([self.labels[i] for i in indices]) #NOTE: Don't use torch.tensor([some python list]) since that only works for 1D lists.
        self.graphs = [self.graphs[i] for i in indices]
    
    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return self.num_classes
