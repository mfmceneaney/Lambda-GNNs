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

# Optuna imports
import optuna
from optuna.samplers import TPESampler

# Fitting imports
import scipy.optimize as opt
from scipy.stats import crystalball

# Utility Imports
import datetime, os, itertools

# Local Imports
from models import GIN, HeteroGIN, Classifier, Discriminator, MLP, Concatenate

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
    verbose=True
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

    # Logs for matplotlib plots
    logs={'train':{'loss':[],'accuracy':[],'roc_auc':[]}, 'val':{'loss':[],'accuracy':[],'roc_auc':[]}}

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
        loss   = criterion(y_pred, y)

        # Cleanup step

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
            loss   = criterion(y_pred, y)

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
    loss.attach(trainer, 'loss')
    # roc_auc   = ROC_AUC(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    # roc_auc.attach(trainer,'roc_auc')

    # Create evaluator
    evaluator = Engine(val_step)

    # Add evaluation metrics
    accuracy_  = Accuracy(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    accuracy_.attach(evaluator, 'accuracy')
    loss_      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
    loss_.attach(evaluator, 'loss')
    # roc_auc_   = ROC_AUC(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    # roc_auc_.attach(evaluator,'roc_auc')

    # Set up early stopping
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
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

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)
    tb_logger.close() #IMPORTANT!
    if save_path!="":
        torch.save(model.to('cpu').state_dict(), os.path.join(log_dir,save_path+"_weights")) #NOTE: Save to cpu state so you can test more easily.
        # torch.save(model.to('cpu'), os.path.join(log_dir,save_path)) #NOTE: Save to cpu state so you can test more easily.
   
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
    f.savefig(os.path.join(log_dir,'training_metrics_loss_'+datetime.datetime.now().strftime("%F")+'.pdf'))

    # Create training/validation accuracy plot
    f = plt.figure()
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['accuracy'],label="training")
    plt.plot(logs['val']['accuracy'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    f.savefig(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+'.pdf'))

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
        dom_y = discriminator(h.detach()) #NOTE: Detach ensures you only propagate gradients to the discriminator, not the model.
        dom_labels = torch.cat([torch.ones(nLabelled,dtype=torch.long), torch.zeros(nUnlabelled,dtype=torch.long)], dim=0).to(device) #NOTE: Make sure domain label lengths match actual batches at the end.
        # dom_labels = torch.cat([torch.cat([torch.ones(nLabelled,1),torch.zeros(nLabelled,1)],dim=1),torch.cat([torch.zeros(nUnlabelled,1),torch.ones(nUnlabelled,1)],dim=1)],dim=0).to(device)
        dom_loss = dom_criterion(dom_y, dom_labels) #NOTE: Using activation function like nn.Sigmoid() at end of model is important since the predictions need to be in [0,1].
        
        dom_loss.backward()
        discriminator_optimizer.step()
        discriminator.zero_grad()
        
        # Step the classifier on training data
        train_y = classifier(h[:nLabelled]) #NOTE: Only train on labelled (i.e., training) data, not domain data.
        dom_y = discriminator(h)
        del h #NOTE: CLEANUP STEP: DEBUGGING
        train_loss = train_criterion(train_y, train_labels)
        dom_loss   = dom_criterion(dom_y, dom_labels) #NOTE: Using nn.Sigmoid() is important since the predictions need to be in [0,1].

        # Get total loss using lambda coefficient for epoch
        # coeff = alpha(engine.state.epoch, max_epochs)#OLD: 7/25/22
        tot_loss = train_loss - alpha * dom_loss
        
        # Step total loss
        tot_loss.backward()
        
        # Step classifier and model optimizer (backwards)
        # classifier_optimizer.step()
        model_optimizer.step() #NOTE: IMPORTANT! This should wrap both classifier and feature extractor parameters.

        # Zero gradients in all parts of model
        model.zero_grad()
        classifier.zero_grad()
        discriminator.zero_grad()

        # Apply softmax and get accuracy on training data
        train_true_y = train_labels.clone().detach().float().view(-1, 1) #NOTE: Labels for cross entropy loss have to be (N) shaped if input is (N,C) shaped.
        train_probs_y = torch.softmax(train_y, 1)
        train_argmax_y = torch.max(train_probs_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
        train_acc = (train_true_y == train_argmax_y.float()).sum().item() / len(train_true_y)

        # Apply softmax and get accuracy on domain data
        dom_true_y = dom_labels.clone().detach().float().view(-1, 1)
        # dom_probs_y = torch.softmax(dom_y, 1) #NOTE: Activation should already be a part of the discriminator
        dom_argmax_y = torch.max(dom_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
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
            # dom_probs_y = torch.softmax(dom_y, 1) #NOTE: Activation should already be a part of the discriminator
            dom_argmax_y = torch.max(dom_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
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
    f = plt.figure()
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
    f.savefig(os.path.join(log_dir,'training_metrics_loss_'+datetime.datetime.now().strftime("%F")+'.pdf'))

    # Create training/validation accuracy plot
    f = plt.figure()
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['train_accuracy'],'-',color='blue',label="classifier training")
    plt.plot(logs['val']['train_accuracy'],'-',color='purple',label="classifier validation")
    plt.plot(logs['train']['dom_accuracy'],'--',color='blue',label="discriminator training")
    plt.plot(logs['val']['dom_accuracy'],'--',color='purple',label="discriminator validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    f.savefig(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+'.pdf'))

    return logs
    
def evaluate(model,device,eval_loader=None,dataset="", prefix="", split=1.0, max_events=1e20, log_dir="logs/",verbose=True, batch_size=32,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=0
        ):
    #TODO: Update args defaults for split and max_events!

    #TODO: Make these options...
    plt.rc('font', size=20) #controls default text size                                                                                                                     
    plt.rc('axes', titlesize=25) #fontsize of the title                                                                                                                     
    plt.rc('axes', labelsize=25) #fontsize of the x and y labels                                                                                                            
    plt.rc('xtick', labelsize=20) #fontsize of the x tick labels                                                                                                            
    plt.rc('ytick', labelsize=20) #fontsize of the y tick labels                                                                                                            
    plt.rc('legend', fontsize=20) #fontsize of the legend

    # plt.rc('font',**{'family':'serif','serif':['Times New Roman']})
    # plt.rc('text', usetex=True)

    figsize=(16,10)

    # Load validation data
    test_dataset = GraphDataset(prefix+dataset) if eval_loader is None else eval_loader.dataset # Make sure this is copied into ~/.dgl folder
    if eval_loader is None:
        test_dataset.load()
        test_dataset = Subset(test_dataset,range(int(min(len(test_dataset),max_events)*split)))

    model.eval()
    model      = model.to(device)

    try:
        model.models = [m.to(device) for m in model.models]
    except Exception as e:
        print(e)

    # Create Dataloader
    dl = GraphDataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    # Loop dataloader and concatenate results
    prediction = None
    test_Y = None
    for x, y in dl:
        x = x.to(device)
        # y = y.to(device) #NOTE: No Need to copy to device
        pred = model(x)
        if prediction is None:
            prediction = pred.clone().detach()
            test_Y = y[:,0].clone().detach()
        else:
            prediction = torch.concatenate((prediction,pred.clone().detach()),axis=0)
            test_Y = torch.concatenate((test_Y,y[:,0].clone().detach()),axis=0)

    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)

    # Copy arrays back to CPU
    test_Y   = test_Y.cpu().view(-1,1) #NOTE: THIS IS NECESSARY SO YOU GET SAME DIMENSION AS argmax_Y BELOW.
    probs_Y  = probs_Y.cpu()
    argmax_Y = argmax_Y.cpu()

    test_acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
    if verbose: print('Accuracy of predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

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

    ##################################################
    # Define fit function
    def func(x, N, beta, m, loc, scale, A, B, C):
        return N*crystalball.pdf(-x, beta, m, -loc, scale) + A*(1 - B*(x - C)**2)
        
    def sig(x, N, beta, m, loc, scale):
        return N*crystalball.pdf(-x, beta, m, -loc, scale)
        
    def bg(x, A, B, C):
        return A*(1 - B*(x - C)**2)
    ##################################################

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
    f = plt.figure(figsize=figsize)
    plt.title('Separated mass distribution')
    hdata = plt.hist(mass_sig_Y[~mass_sig_Y.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    
    # Fit output of NN
    print("DEBUGGING: hdata[0] = ",hdata[0])#DEBUGGING
    N, beta, m, loc, scale, A, B, C = 500, 1, 1.112, 1.115, 0.008, hdata[0][-1], 37, 1.24 #OLD: A = hdata[0][-1] #DEBUGGING COMMENTED OUT!
    if A == 0: A = 0.1 #DEBUGGING
    d_N, d_beta, d_m, d_loc, d_scale, d_A, d_B, d_C = N/0.01, beta/0.1, m/0.1, loc/0.1, scale/0.01, A/10, B/0.1, C/0.1
    parsMin = [N-d_N, beta-d_beta, m-d_m, loc-d_loc, scale-d_scale, A-d_A, B-d_B, C-d_C]
    parsMax = [N+d_N, beta+d_beta, m+d_m, loc+d_loc, scale+d_scale, A+d_A, B+d_B, C+d_C]
    print("DEBUGGING: parsMin = ",parsMin)#DEBUGGING
    print("DEBUGGING: parsMax = ",parsMax)#DEBUGGING
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
    lg += f"N = {round(optParams[0],0)}±{round(pcov[0,0],4)}\n"
    lg += f"α = {round(optParams[1],3)}±{round(pcov[1,1],7)}\n"
    lg += f"n = {round(optParams[2],3)}±{round(pcov[2,2],2)}\n"
    lg += f"μ = {round(optParams[3],5)}±{round(pcov[3,3],10)}\n"
    lg += f"σ = {round(optParams[4],5)}±{round(pcov[4,4],10)}\n"
    lg += f"A = {round(optParams[5],0)}±{round(pcov[5,5],2)}\n"
    lg += f"β = {round(optParams[6],0)}±{round(pcov[6,6],2)}\n"
    lg += f"M = {round(optParams[7],2)}±{round(pcov[7,7],7)}\n"
    plt.text(low_high[1]-(low_high[1]-low_high[0])/3,1/2*max(hdata[0]),lg,fontsize=20,linespacing=1.25) #NOTE: MAKE THESE PARAMS OPTIONS

    # Show the graph
    # plt.hist(mass_bg_Y[~mass_bg_Y.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background') #NOTE: COMMENTED OUT FOR DEBUGGING
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'eval_metrics_mass_'+datetime.datetime.now().strftime("%F")+'.pdf'))

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
    f.savefig(os.path.join(log_dir,'test_metrics_mass_'+datetime.datetime.now().strftime("%F")+'.pdf'))

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
    f.savefig(os.path.join(log_dir,'test_metrics_mass_true_'+datetime.datetime.now().strftime("%F")+'.pdf'))

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
    f.savefig(os.path.join(log_dir,'test_metrics_mass_false_'+datetime.datetime.now().strftime("%F")+'.pdf'))

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
    f.savefig(os.path.join(log_dir,'mc_matched_mass_'+datetime.datetime.now().strftime("%F")+'.pdf'))

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
    f.savefig(os.path.join(log_dir,'mc_matched_nn_sig_mass_'+datetime.datetime.now().strftime("%F")+'.pdf'))

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
    f.savefig(os.path.join(log_dir,'mc_matched_nn_bg_mass_'+datetime.datetime.now().strftime("%F")+'.pdf'))

    #-----#
    #NOTE: ADDED DEBUGGING

    # pid, fraction =  2 5 %
    # pid, fraction =  3 0 %
    # pid, fraction =  91 1 %
    # pid, fraction =  92 74 %
    # pid, fraction =  2212 0 %
    # pid, fraction =  3114 1 %
    # pid, fraction =  3212 8 %
    # pid, fraction =  3214 3 %
    # pid, fraction =  3224 3 %

    unique_ppa_pids = np.unique(
                        test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,-1].clone().detach().float(),
                        return_counts=True #NOTE: IMPORTANT!
                        )
    available_labels = unique_ppa_pids[0]
    counts_labels    = unique_ppa_pids[1]
    print("DEBUGGING: LABEL COUNTS     = ",[int(el) for el in counts_labels])#DEBUGGING
    print("DEBUGGING: AVAILABLE LABELS = ",[int(el) for el in available_labels])#DEBUGGING

    labels, counts = available_labels, counts_labels
    labels = [int(el) for el in labels]
    print("labels = ",labels)#DEBUGGING
    print("counts = ",counts)#DEBUGGING
    total = np.sum(counts)
    print("total = ",total)#DEBUGGING
    bank = {labels[idx]:round(el,4) for idx, el in enumerate(counts/total)}
    for el in bank.keys():
        print("pid, fraction = ",el,int(bank[el]*100),"%")

    x_multi_sig_true  = [] #NOTE: APPEND ABOVE ONLY IF COUNTS/TOTAL>1%
    x_multi_sig_false = []
    x_multi_bg_true   = []
    x_multi_bg_false  = []
    labels_sig_true   = [] #NOTE: APPEND ABOVE ONLY IF COUNTS/TOTAL>1%
    labels_sig_false  = []
    labels_bg_true    = [] 
    labels_bg_false   = []

    name_bank = {
        1 : "d",
        2 : "u",
        3 : "s",
        91 : "cluster (91)",
        92 : "string (92)",
        2212 : "p",
        3224 : "$\Sigma^{*+}$",
        3214 : "$\Sigma^{*0}$",
        3212 : "$\Sigma^{0}$",
        3114 : "$\Sigma^{*-}$",
        3312 : "$\Xi^{-}$",
        3322 : "$\Xi^{0}$"
    }

    for my_pid__ in bank.keys():
        fill_option = False
        if bank[my_pid__]>0.01:
            fill_option = True

        mass_sig_true_from_target  = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),
                                    mask=np.logical_or(
                                        ~(my_pid__ == test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,-1].clone().detach().float()),
                                        np.logical_or(
                                            ~(torch.squeeze(argmax_Y) == 1),
                                            ~(torch.squeeze(argmax_Y) == test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float())
                                        )
                                    )
                                )

        mass_sig_false_from_target  = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),
                                    mask=np.logical_or(
                                        ~(my_pid__ == test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,-1].clone().detach().float()),
                                        np.logical_or(
                                            ~(torch.squeeze(argmax_Y) == 1),
                                            ~(torch.squeeze(argmax_Y) != test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float())
                                        )
                                    )
                                )

        # # Plot MC-Matched distributions for NN-identified signal
        # bins = 100
        # # low_high = (1.1,1.13)
        # f = plt.figure(figsize=figsize)
        # plt.title(f"Lambda Parent PID {my_pid__:.0f} NN-identified signal mass distribution MC-matched")
        # plt.hist(mass_sig_true_from_target[~mass_sig_true_from_target.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='true')
        # plt.hist(mass_sig_false_from_target[~mass_sig_false_from_target.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='false')
        # plt.legend(loc='upper left', frameon=False)
        # plt.ylabel('Counts')
        # plt.xlabel('Invariant mass (GeV)')
        # f.savefig(os.path.join(log_dir,f"TEST_ppa_pid_MC_{my_pid__:.0f}__mc_matched_nn_sig_mass_"+datetime.datetime.now().strftime("%F")+'.pdf'))


        #NOTE: NOW LOOK AT NN-IDENTIFIED BG

        mass_bg_true_from_target  = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),
                                    mask=np.logical_or(
                                        ~(my_pid__ == test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,-1].clone().detach().float()),
                                        np.logical_or(
                                            ~(torch.squeeze(argmax_Y) == 0),
                                            ~(torch.squeeze(argmax_Y) == test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float())
                                        )
                                    )
                                )

        mass_bg_false_from_target  = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),
                                    mask=np.logical_or(
                                        ~(my_pid__ == test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,-1].clone().detach().float()),
                                        np.logical_or(
                                            ~(torch.squeeze(argmax_Y) == 0),
                                            ~(torch.squeeze(argmax_Y) != test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float())
                                        )
                                    )
                                )

        # # Plot MC-Matched distributions for NN-identified signal
        # bins = 100
        # # low_high = (1.1,1.13)
        # f = plt.figure(figsize=figsize)
        # plt.title(f"Proton Parent Parent PID {my_pid__:.0f} NN-identified background mass distribution MC-matched")
        # plt.hist(mass_bg_true_from_target[~mass_bg_true_from_target.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='true')
        # plt.hist(mass_bg_false_from_target[~mass_bg_false_from_target.mask], color='c', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='false')
        # plt.legend(loc='upper left', frameon=False)
        # plt.ylabel('Counts')
        # plt.xlabel('Invariant mass (GeV)')
        # f.savefig(os.path.join(log_dir,f"TEST_ppa_pid_MC_{my_pid__:.0f}__mc_matched_nn_bg_mass_"+datetime.datetime.now().strftime("%F")+'.pdf'))

        if fill_option:
            x_multi_sig_true.append(mass_sig_true_from_target[~mass_sig_true_from_target.mask]) #NOTE: APPEND ABOVE ONLY IF COUNTS/TOTAL>1%
            x_multi_sig_false.append(mass_sig_false_from_target[~mass_sig_false_from_target.mask])
            x_multi_bg_true.append(mass_bg_true_from_target[~mass_bg_true_from_target.mask])
            x_multi_bg_false.append(mass_bg_false_from_target[~mass_bg_false_from_target.mask])
            newname = "noname"
            try:
                newname = name_bank[my_pid__]
            except KeyError as e:
                print(e)
                newname = str(my_pid__)

            labels_sig_true.append(newname) #NOTE: APPEND ABOVE ONLY IF COUNTS/TOTAL>1%
            labels_sig_false.append(newname)
            labels_bg_true.append(newname)
            labels_bg_false.append(newname)

    # RESHUFFLE DATASETS TO CONTRIBUTIONS ARE PLOTTED SMALLEST TO LARGEST
    def reorder_(x,labels):

        idcs_ = {len(x[idx]):idx for idx in range(len(x))} #NOTE: ALL SHOULD HAVE SAME LENGTH???
        idcs = []
        print("---------------------------------------------")
        print("idcs_.keys() = ",idcs_.keys())#DEBUGGING
        print("np.sort(list(idcs_.keys())) = ",np.sort(list(idcs_.keys())))#DEBUGGING
        for idx in np.sort(list(idcs_.keys())): #NOTE: SORT BY COUNTS, HERE SMALLEST COUNTS ADDED FIRST
            idcs.append(idcs_[idx])
        print("DEBUGGING:AFTER idcs_ = ",idcs_)#DEBUGGING
        print("DEBUGGING:AFTER idcs = ",idcs)#DEBUGGING
        x       = [x[idx]       for idx in idcs]
        labels  = [labels[idx]  for idx in idcs]

        return x, labels

    x_multi_sig_true, labels_sig_true   = reorder_(x_multi_sig_true, labels_sig_true)
    x_multi_sig_false, labels_sig_false = reorder_(x_multi_sig_false, labels_sig_false)
    x_multi_bg_true, labels_bg_true     = reorder_(x_multi_bg_true, labels_bg_true )
    x_multi_bg_false, labels_bg_false   = reorder_(x_multi_bg_false, labels_bg_false)

    # x_multi_sig_true  = [x_multi_sig_true[idx]  for idx in idcs]
    # x_multi_sig_false = [x_multi_sig_false[idx] for idx in idcs]
    # x_multi_bg_true   = [x_multi_bg_true[idx]   for idx in idcs]
    # x_multi_bg_false  = [x_multi_bg_false[idx]  for idx in idcs]

    # labels_sig_true  = [labels_sig_true[idx]  for idx in idcs]
    # labels_sig_false = [labels_sig_false[idx] for idx in idcs]
    # labels_bg_true   = [labels_bg_true[idx]   for idx in idcs]
    # labels_bg_false  = [labels_bg_false[idx]  for idx in idcs]


    print("DEBUGGING: labels_sig_true = ",labels_sig_true)#DEBUGGING
    print("DEBUGGING: labels_sig_false = ",labels_sig_false)#DEBUGGING
    print("DEBUGGING: labels_bg_true = ",labels_bg_true)#DEBUGGING
    print("DEBUGGING: labels_bg_false = ",labels_bg_false)#DEBUGGING

    print("DEBUGGING: x_multi_sig_true")#DEBUGGING
    for el in x_multi_sig_true:
        print("\tnp.shape(el) = ",np.shape(el))#DEBUGGING

    # PLOT FULL SPECTRUM SIG/BG TRUE/FALSE
    # Make a multiple-histogram of data-sets with different length.
    f = plt.figure(figsize=figsize)
    x_multi_sig = [mass_sig_false[~mass_sig_false.mask],mass_sig_true[~mass_sig_true.mask]]
    x_multi_bg  = [mass_bg_true[~mass_bg_true.mask], mass_bg_false[~mass_bg_false.mask]]
    plt.title('NN-Identified Mass Spectrum Proton Parent Parent Decomposition')
    plt.hist(x_multi_sig, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=('sig_false','sig_true')) #NOTE: MAKE SURE THESE MATCH UP!!!
    plt.hist(x_multi_bg, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=('bg_true','bg_false')) #NOTE: ADD BG FIRST SO FALSE SIGNAL IS VISIBLE IN CORRECT COLOR

    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,"parent_decomposition_full.pdf"))

    # SIG BOTH TRUE/FALSE
    # Make a multiple-histogram of data-sets with different length.
    f = plt.figure(figsize=figsize)
    plt.title('NN-Identified Signal Proton Parent Parent Decomposition')
    plt.hist(x_multi_sig_true, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=["Sig true "+name for name in labels_sig_true])
    plt.hist(x_multi_sig_false, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=["Sig false "+name for name in labels_sig_false])
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,"parent_decomposition_sig.pdf"))
    
    # BG BOTH TRUE/FALSE
    # Make a multiple-histogram of data-sets with different length.
    f = plt.figure(figsize=figsize)
    plt.title('NN-Identified Background Proton Parent Parent Decomposition')
    plt.hist(x_multi_bg_true, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=["Bg true "+name for name in labels_bg_true])
    plt.hist(x_multi_bg_false, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=["Bg false "+name for name in labels_bg_false])
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,"parent_decomposition_bg.pdf"))

    # JUST SIG TRUE
    # Make a multiple-histogram of data-sets with different length.
    f = plt.figure(figsize=figsize)
    plt.title('NN-Identified True Signal $\Lambda$ Parent Decomposition')
    plt.hist(x_multi_sig_true, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=labels_sig_true)
    # plt.hist(x_multi_sig_false, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=labels_sig_false)
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,"parent_decomposition_sig_true.pdf"))

    # JUST SIG FALSE
    # Make a multiple-histogram of data-sets with different length.
    f = plt.figure(figsize=figsize)
    plt.title('NN-Identified False Signal Proton Parent Parent Decomposition')
    # plt.hist(x_multi_sig_true, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=labels_sig_true)
    plt.hist(x_multi_sig_false, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=labels_sig_false)
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,"parent_decomposition_sig_false.pdf"))

    # JUST BG TRUE
    # Make a multiple-histogram of data-sets with different length.
    f = plt.figure(figsize=figsize)
    plt.title('NN-Identified True Background Proton Parent Parent Decomposition')
    plt.hist(x_multi_bg_true, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=labels_bg_true)
    # plt.hist(x_multi_bg_false, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=labels_bg_false)
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,"parent_decomposition_bg_true.pdf"))

    # JUST BG FALSE
    # Make a multiple-histogram of data-sets with different length.
    f = plt.figure(figsize=figsize)
    plt.title('NN-Identified False Background $\Lambda$ Parent Decomposition')
    # plt.hist(x_multi_bg_true, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=labels_bg_true)
    plt.hist(x_multi_bg_false, bins=bins, range=low_high, alpha=0.5, histtype='stepfilled', stacked=True, density=False, label=labels_bg_false)
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,"parent_decomposition_bg_false.pdf"))

    #-----#

    print("DEBUGGING: np.unique(test_Y.detach().numpy()) = ",np.unique(test_Y.detach().numpy()))#DEBUGGING

    # Get ROC curve
    pfn_fp, pfn_tp, threshs = roc_curve(test_Y.detach().numpy(), probs_Y[:,1].detach().numpy())

    # Get area under the ROC curve

    auc = roc_auc_score(np.squeeze(test_Y.detach().numpy()), probs_Y[:,1].detach().numpy())
    if verbose: print(f'AUC = {auc:.4f}')
    if verbose: print(f'test_acc = {test_acc:.4f}')#DEBUGGING ADDED

    # Create matplotlib plots for ROC curve and testing decisions
    f = plt.figure(figsize=figsize)

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
    f.savefig(os.path.join(log_dir,model.name+"_ROC_"+datetime.datetime.now().strftime("%F")+".pdf"))

    ##########################################################
    # Plot testing decisions
    bins = 100
    low = min(np.min(p) for p in probs_Y[:,1].detach().numpy())
    high = max(np.max(p) for p in probs_Y[:,0].detach().numpy())
    low_high = (low,high)
    f = plt.figure(figsize=figsize)
    plt.clf()
    plt.hist(probs_Y[:,1][torch.squeeze(torch.logical_and(test_Y==1,argmax_Y==1))].detach().numpy(), color='tab:red', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='True signal')
    plt.hist(probs_Y[:,1][torch.squeeze(torch.logical_and(test_Y==0,argmax_Y==1))].detach().numpy(), color='tab:orange', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='False signal')
    plt.hist(probs_Y[:,1][torch.squeeze(torch.logical_and(test_Y==1,argmax_Y==0))].detach().numpy(), color='tab:green', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='False background')
    plt.hist(probs_Y[:,1][torch.squeeze(torch.logical_and(test_Y==0,argmax_Y==0))].detach().numpy(), color='tab:blue', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='True background')
    plt.xlabel('output')
    plt.ylabel('counts')
    plt.yscale('log')
    f.savefig(os.path.join(log_dir,model.name+"_test_decisions_"+datetime.datetime.now().strftime("%F")+".pdf"))

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

        # Now treat case that validation_dataset is specified
        if args.validation_dataset is not None:
            if len(args.indices)>3:
                _, val_dataloader, eval_loader, _, _, _ = load_graph_dataset(dataset=args.validation_dataset, prefix=args.prefix, 
                                                        split=args.split, max_events=args.max_events, indices=args.indices,
                                                        num_workers=args.nworkers, batch_size=batch_size) 
            elif len(args.indices)==3:
                _, val_dataloader, _, _, _ = load_graph_dataset(dataset=args.validation_dataset, prefix=args.prefix, 
                                                        split=args.split, max_events=args.max_events, indices=args.indices,
                                                        num_workers=args.nworkers, batch_size=batch_size)
            else:
                _, val_dataloader, _, _, _ = load_graph_dataset(dataset=args.validation_dataset, prefix=args.prefix, 
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
        trialdir = args.study_name+'_trial_'+str(trial.number+1)
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
            verbose=args.verbose
        )

        #NOTE: #TODO: add extra args to optimize script parser and save hyperparameter choices in running script???? Or just use mlflow....
        # evaluate_on_data(
        #     model,
        #     device,
        #     dataset=args.datadataset,
        #     prefix=args.prefix,
        #     split=args.split,
        #     log_dir=trialdir,
        #     verbose=args.verbose
        # )

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
        alpha = args.alpha[0] if args.alpha[0] == args.alpha[1] else trial.suggest_int("alpha",args.alpha[0],args.alpha[1])
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

        # Now treat case that validation_dataset is specified
        if args.validation_dataset is not None:
            if len(args.indices)>3:
                _, val_dataloader, eval_loader, _, _, _ = load_graph_dataset(dataset=args.validation_dataset, prefix=args.prefix, 
                                                        split=args.split, max_events=args.max_events, indices=args.indices,
                                                        num_workers=args.nworkers, batch_size=batch_size) 
            elif len(args.indices)==3:
                _, val_dataloader, _, _, _ = load_graph_dataset(dataset=args.validation_dataset, prefix=args.prefix, 
                                                        split=args.split, max_events=args.max_events, indices=args.indices,
                                                        num_workers=args.nworkers, batch_size=batch_size)
            else:
                _, val_dataloader, _, _, _ = load_graph_dataset(dataset=args.validation_dataset, prefix=args.prefix, 
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
        classifier = MLP(nmlp_head, hdim, hdim_head, nclasses).to(device)
        discriminator = MLP(nmlp_head, hdim, hdim_head, n_domains).to(device)

        # # Make models parallel if multiple gpus available
        # if device.type=='cuda' and device.index==None:
        #     model = DataParallel(model)
        #     classifier = DataParallel(classifier)
        #     discriminator = DataParallel(discriminator)

        # Create optimizers
        model_optimizer = optim.Adam(list(model.parameters())+list(classifier.parameters()), lr=lr)
        classifier_optimizer = None #optim.Adam(classifier.parameters(), lr=lr_c)#NOTE: This is now extraneous.
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
        trialdir = args.study_name+'_trial_'+str(trial.number+1)
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

def evaluate_on_data(model,device,dataset="", prefix="", split=1.0, log_dir="logs/",verbose=True,batch_size=32,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=0
        ):

    #TODO: Make these options...
    plt.rc('font', size=20) #controls default text size                                                                                                                     
    plt.rc('axes', titlesize=25) #fontsize of the title                                                                                                                     
    plt.rc('axes', labelsize=25) #fontsize of the x and y labels                                                                                                            
    plt.rc('xtick', labelsize=20) #fontsize of the x tick labels                                                                                                            
    plt.rc('ytick', labelsize=20) #fontsize of the y tick labels                                                                                                            
    plt.rc('legend', fontsize=15) #fontsize of the legend

    # Load validation data
    test_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    test_dataset.load()
    test_dataset = Subset(test_dataset,range(int(len(test_dataset)*split)))

    model.eval()
    model      = model.to(device)
    try:
        model.models = [m.to(device) for m in model.models]
        for m in model.models:
            m.eval()
    except Exception as e:
        print(e)
    # test_bg    = dgl.batch(test_dataset.dataset.graphs[test_dataset.indices.start:test_dataset.indices.stop])#TODO: Figure out nicer way to use subset
    # test_bg    = test_bg.to(device)

    # Create Dataloader
    dl = GraphDataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    # Loop dataloader and concatenate results
    prediction = None
    for x, y in dl:
        x = x.to(device)
        # y = y.to(device) #NOTE: Unnecessary for data
        pred = model(x)
        if prediction is None:
            prediction = pred.clone().detach()
        else:
            prediction = torch.concatenate((prediction,pred.clone().detach()),axis=0)

    # Get probabilities
    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)

    # Copy arrays back to CPU
    probs_Y  = probs_Y.cpu()
    argmax_Y = argmax_Y.cpu()

    print("DEBUGGING: test_dataset.indices.start = ",test_dataset.indices.start)#DEBUGGING
    print("DEBUGGING: test_dataset.indices.stop = ",test_dataset.indices.stop)#DEBUGGING
    print("DEBUGGING: np.min(ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float()])) = ",np.min( ma.array( test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float() )))#DEBUGGING

    # Get separated mass distributions
    mass_sig_Y    = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),mask=~(argmax_Y == 1))
    mass_bg_Y     = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float(),mask=~(argmax_Y == 0))

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
    # mass_all = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,1].clone().detach().float())
    # hdata = plt.hist(mass_all, color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    hdata = plt.hist(mass_sig_Y[~mass_sig_Y.mask], color='m', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    # plt.show()#DEBUGGING
    # return #DEBUGGING
    
    # Fit output of NN
    print("DEBUGGING: hdata[0][-1] = ",hdata[0][-1])#DEBUGGING
    N, beta, m, loc, scale, A, B, C = 5, 1, 1.112, 1.115, 0.008, hdata[0][-1], 37, 1.24
    if A==0: A = 0.1#DEBUGGING
    d_N, d_beta, d_m, d_loc, d_scale, d_A, d_B, d_C = N/0.01, beta/0.1, m/0.1, loc/0.1, scale/0.01, A/10, B/0.1, C/0.1
    #d_N, d_beta, d_m, d_loc, d_scale, d_A, d_B, d_C = N/0.001, beta/0.001, m/0.01, loc/0.001, scale/0.001, A/1, B/0.1, C/5
    parsMin = [N-d_N, beta-d_beta, m-d_m, loc-d_loc, scale-d_scale, A-d_A, B-d_B, C-d_C]
    parsMax = [N+d_N, beta+d_beta, m+d_m, loc+d_loc, scale+d_scale, A+d_A, B+d_B, C+d_C]
    print(parsMin)#DEBUGGING
    print(parsMax)#DEBUGGING
    optParams, pcov = opt.curve_fit(func, hdata[1][:-1], hdata[0], method='trf', bounds=(parsMin,parsMax))

    # Plot fit
    x = np.linspace(low_high[0],low_high[1],bins)#mass_sig_Y[~mass_sig_Y.mask]
    y = hdata[0]
    plt.plot(x, func(x, *optParams), color='r')
    plt.plot(x, sig(x, *optParams[0:5]), color='tab:purple')
    plt.plot(x, bg(x, *optParams[5:]), color='b')
    bghist = plt.hist(x, weights=y-bg(x, *optParams[5:]), bins=bins, range=low_high, histtype='step', alpha=0.5, color='b')
    
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

    integral_bghist = sum(bghist[0][bin1:bin2])*binwidth
    print("integral_bghist = ",integral_bghist)#DEBUGGING

    print("optParams = ",optParams)#DEBUGGING

    resultN = integrate.quad(lambda x: func(x, *optParams),mmin,mmax)[0] / binwidth
    resultS = integrate.quad(lambda x: sig(x, *optParams[0:5]),mmin,mmax)[0] / binwidth
    resultB = integrate.quad(lambda x: bg(x, *optParams[5:]),mmin,mmax)[0] / binwidth

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
    f.savefig(os.path.join(log_dir,'eval_metrics_mass_'+datetime.datetime.now().strftime("%F")+'.pdf'))

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
    plt.yscale('log')
    f.savefig(os.path.join(log_dir,model.name+"_eval_decisions_"+datetime.datetime.now().strftime("%F")+".pdf"))

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
