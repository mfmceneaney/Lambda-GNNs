###############################
# Matthew McEneaney
# COLABORATORY VERSION
# 7/29/21
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

# PyTorch Ignite Imports
from ignite.engine import Engine, Events, EventEnum, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import global_step_from_engine, EarlyStopping

# Optuna imports
import optuna
from optuna.samplers import TPESampler

# Utility Imports
import datetime, os

# Local Imports
from models import GIN, HeteroGIN

def load_graph_dataset(dataset="ldata_6_22",prefix="",batch_size=1024,drop_last=False,shuffle=True,num_workers=0,pin_memory=True, verbose=True):
    
    # Add directory prefix for colab
    dataset = os.path.join(prefix,dataset)

    # Load training data
    train_dataset = LambdasDataset(dataset+"_train") # Make sure this is copied into ~/.dgl folder
    train_dataset.load()
    num_labels = train_dataset.num_labels
    node_feature_dim = train_dataset.graphs[0].ndata["data"].shape[-1]

    # Create training dataloader
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)

    # Load validation data
    val_dataset = LambdasDataset(dataset+"_test") # Make sure this is copied into ~/.dgl folder
    val_dataset.load()

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
            dataset="ldata_6_22",prefix="", log_interval=10,log_dir="logs/",save_path="torch_models",verbose=True):

    # Make sure log/save directories exist
    try:
        os.mkdir(log_dir+"tb_logs/tmp") #NOTE: Do NOT use os.path.join() here since it requires that the directory exist.
    except Exception:
        if verbose: print("Could not create directory:",os.path.join(log_dir,"tb_logs/tmp"))

    # Show model if requested
    if verbose: print(model)

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
        y_pred = model(x)#TODO: Modify this so it works with PFN/EFN as well?
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
        prev_lr = scheduler.get_last_lr()
        scheduler.step()
        new_lr = scheduler.get_last_lr()
        if prev_lr != new_lr and verbose:
            print(f"\nLearning rate: {prev_lr[0]:.4f} -> {new_lr[0]:.4f}",end="")

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
        torch.save(model.state_dict(), save_path)

    # # Create training/validation loss plot
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

    # # Create training/validation accuracy plot
    f = plt.figure()
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['accuracy'],label="training")
    plt.plot(logs['val']['accuracy'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    f.savefig(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))
    
    # if verbose: plt.show() #DEBUGGING...for now...

def evaluate(model,device,dataset="ldata_6_22",prefix="",log_dir="logs/",verbose=True):
    #TODO: Add .to(device) for this method so the argument isn't useless
    # Load validation data
    test_dataset = LambdasDataset(os.path.join(prefix,dataset+"_test")) # Make sure this is copied into ~/.dgl folder
    test_dataset.load()

    model.eval()
    test_bg    = batch(test_dataset.graphs)
    test_Y     = test_dataset.labels[:,0].clone().detach().float().view(-1, 1) #IMPORTANT: keep .view() here
    test_bg    = test_bg.to(device)
    test_Y     = test_Y.to(device)
    prediction = model(test_bg)
    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)
    if verbose: print('Accuracy of predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

    # Get separated mass distributions
    print(test_dataset.labels[:,1].clone().detach().float().shape)
    print(argmax_Y.shape)
    print(argmax_Y.device)
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
    f.savefig(os.path.join(log_dir,'test_metrics_mass_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

    # Plot correct mass decisions separated into signal/background
    bins = 100
    low_high = (1.1,1.13)
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
    low_high = (1.1,1.13)
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
    low_high = (1.1,1.13)
    f = plt.figure()
    plt.title('Separated mass distribution MC-matched')
    plt.hist(mass_sig_MC[~mass_sig_MC.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
    plt.hist(mass_bg_MC[~mass_bg_MC.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')
    f.savefig(os.path.join(log_dir,'mc_matched_mass_'+datetime.datetime.now().strftime("%F")+dataset+'.png'))

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
    test_dataset = LambdasDataset(os.path.join(args.prefix,args.dataset+"_test")) # Make sure this is copied into ~/.dgl folder
    test_dataset.load()

    def objective(trial):

        # Get parameter suggestions for trial
        batch_size = trial.suggest_int("batch_size",args.batch[0],args.batch[1])
        nlayers = trial.suggest_int("nlayers",args.nlayers[0],args.nlayers[1])
        nmlp  = trial.suggest_int("nmlp",args.nmlp[0],args.nmlp[1])
        hdim  = trial.suggest_int("hdim",args.hdim[0],args.hdim[1])
        do    = trial.suggest_float("do",args.dropout[0],args.dropout[1])
        lr    = trial.suggest_float("lr",args.lr[0],args.lr[1],log=True)#TODO: Not sure about log yet...
        step  = trial.suggest_int("step",args.step[0],args.step[1])
        gamma = trial.suggest_float("gamma",args.gamma[0],args.gamma[1])
        max_epochs = args.epochs

        # Setup data and model
        train_loader, val_loader, nclasses, nfeatures = load_graph_dataset(dataset=args.dataset,prefix=args.prefix,
                                                        num_workers=args.nworkers, batch_size=batch_size)

        # Initiate model and optimizer, scheduler, loss
        model = GIN(nlayers,nmlp,nfeatures,hdim,nclasses,do,args.learn_eps,args.npooling,args.gpooling).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
        criterion = nn.CrossEntropyLoss()

        # Make sure log/save directories exist
        trialdir = 'trial_'+datetime.datetime.now().strftime("%F")+'_'+args.dataset+'_'+str(trial.datetime_start)
        try:
            os.mkdir(args.log+trialdir) #NOTE: Do NOT use os.path.join() here since it requires that the directory already exist.
        except FileExistsError:
            if args.verbose: print("Directory:",os.path.join(args.log,trialdir))
        trialdir = os.path.join(args.log,trialdir)

        # Show model if requested
        if args.verbose: print(model)

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
            y_pred = model(x)#TODO: Modify this so it works with PFN/EFN as well?
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

        @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
        def log_training_loss(trainer):
            if verbose: print(f"\rEpoch[{trainer.state.epoch}/{max_epochs} : " +
                f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
                f"Loss: {trainer.state.output['loss']:.3f} Accuracy: {trainer.state.output['accuracy']:.3f}",end='')

        @trainer.on(Events.EPOCH_COMPLETED)
        def stepLR(trainer):
            prev_lr = scheduler.get_last_lr()
            scheduler.step()
            new_lr = scheduler.get_last_lr()
            if prev_lr != new_lr and verbose:
                print(f"\nLearning rate: {prev_lr[0]:.4f} -> {new_lr[0]:.4f}",end="")

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
            torch.save(model.state_dict(), save_path)#TODO: Make unique identifier by trial?

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
        test_bg    = batch(test_dataset.graphs)
        test_Y     = test_dataset.labels[:,0].clone().detach().float().view(-1, 1) #IMPORTANT: keep .view() here
        prediction = model(test_bg)
        probs_Y    = torch.softmax(prediction, 1)
        argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)
        test_acc   = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
        if verbose: print('Accuracy of predictions on the test set: {:4f}%'.format(
            (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

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
        low_high = (1.1,1.13)
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
        low_high = (1.1,1.13)
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
        low_high = (1.1,1.13)
        f = plt.figure()
        plt.title('Separated mass distribution MC-matched')
        plt.hist(mass_sig_MC[~mass_sig_MC.mask], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')
        plt.hist(mass_bg_MC[~mass_bg_MC.mask], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='background')
        plt.legend(loc='upper left', frameon=False)
        plt.ylabel('Counts')
        plt.xlabel('Invariant mass (GeV)')
        f.savefig(os.path.join(trialdir,'mc_matched_mass_'+datetime.datetime.now().strftime("%F")+args.dataset+'.png'))

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
        f.savefig(os.path.join(trialdir,model.name+"_test_decisions_"+datetime.datetime.now().strftime("%F")+args.dataset+".png"))

        # CLOSE FIGURES: IMPORTANT FOR MEMORY
        plt.close('all')

        return test_acc

    ##### MAIN PART #####

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    sampler = TPESampler() #TODO: Add command line option for selecting different sampler types.
    study = optuna.create_study(sampler=sampler,direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=args.ntrials, timeout=args.timeout)
    trial = study.best_trial

    if verbose:
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

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