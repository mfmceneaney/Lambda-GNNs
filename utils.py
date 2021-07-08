###############################
# Matthew McEneaney
# 6/24/21
###############################

from __future__ import absolute_import, division, print_function

# ML Imports
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib as mpl
import matplotlib.pyplot as plt

# DGL Graph Learning Imports
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_info, load_info

# PyTorch Imports
import torch

# PyTorch Ignite Imports
from ignite.engine import Engine, Events, EventEnum, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import global_step_from_engine
from ignite.contrib.handlers.base_logger import (
    BaseLogger,
    BaseOptimizerParamsHandler,
    BaseOutputHandler,
    BaseWeightsHistHandler,
    BaseWeightsScalarHandler,
)

# Utility Imports
import datetime, os, psutil, threading

def load_graph_dataset(dataset="ldata_train_6_23",batch_size=256,drop_last=False,shuffle=True,num_workers=1,pin_memory=False):
    # Load training data
    train_dataset = LambdasDataset("ldata_train_6_23") # Make sure this is copied into ~/.dgl folder
    train_dataset.load()
    num_labels = train_dataset.num_labels
    node_feature_dim = train_dataset.graphs[0].ndata["data"].shape[0]
    print("*** NODE_FEATURE_DIM *** = ",node_feature_dim)

    # Create training dataloader
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory)

    # Load validation data
    val_dataset = LambdasDataset("ldata_test_6_23") # Make sure this is copied into ~/.dgl folder
    val_dataset.load()

    # Create testing dataloader
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory)

    return train_loader, val_loader, num_labels, node_feature_dim

def train(args, model, device, train_loader, val_loader, optimizer, scheduler, criterion, max_epochs,
            log_interval=10,log_dir="./tb_logs/tmp/",save_path="./torch_models",verbose=True):

    # Make sure log/save directories exist
    try:
        os.mkdir(log_dir)
    except Exception:
        if verbose: print("Could not create directory:",log_dir)

    # Create trainer
    def train_step(engine, batch):
        model.train()
        x, y   = batch
        x      = x.to(device)
        y      = y.to(device)
        y_pred = model(x,x.ndata["data"].float())
        loss   = criterion(y_pred, y)
        acc    = (y_pred.argmax(1) == y).type(torch.float).sum().item() / len(y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(),
                'accuracy': acc,
                'y_pred': y_pred,
                'y': y}

    trainer = Engine(train_step)

    # Add metrics
    accuracy  = Accuracy(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
    accuracy.attach(trainer, 'accuracy')
    loss      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
    loss.attach(trainer, 'loss')
    roc_auc   = ROC_AUC(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
    roc_auc.attach(trainer,'roc_auc')
    roc_curve = RocCurve(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
    roc_curve.attach(trainer,'roc_curve')

    # Create validator
    def val_step(engine, batch):
        model.eval()
        x, y   = batch
        x      = x.to(device)
        y      = y.to(device)
        y_pred = model(x,x.ndata["data"].float())
        loss   = criterion(y_pred, y)
        acc    = (y_pred.argmax(1) == y).type(torch.float).sum().item() / len(y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.train()
        return {'loss': loss.item(),
                'accuracy': acc,
                'y_pred': y_pred,
                'y': y}

    evaluator = Engine(val_step)

    # Add metrics
    accuracy_  = Accuracy(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
    accuracy_.attach(evaluator, 'accuracy')
    loss_      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
    loss_.attach(evaluator, 'loss')
    roc_auc_   = ROC_AUC(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
    roc_auc_.attach(evaluator,'roc_auc')
    roc_curve_ = RocCurve(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
    roc_curve_.attach(evaluator,'roc_curve')

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(trainer):
        if verbose: print(f"\rEpoch[{trainer.state.epoch} : " +
            f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
            f"Loss: {trainer.state.output['loss']:.3f} Accuracy: {trainer.state.output['accuracy']:.3f}",end='')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        metrics = evaluator.run(train_loader).metrics
        logs['train'].append({metric:metrics[metric] for metric in metrics.keys()})
        if verbose: print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.4f} Avg loss: {metrics['loss']:.4f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        metrics = evaluator.run(val_loader).metrics
        logs['val'].append({metric:metrics[metric] for metric in metrics.keys()})
        if verbose: print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.4f} Avg loss: {metrics['loss']:.4f}")

    # Create a TensorBoard logger
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    tb_logger.attach(trainer,event_name=Events.EPOCH_COMPLETED,log_handler=WeightsHistHandler(model))

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"loss": loss["loss"]}
    )
        
    # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
    # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
    # of the `trainer` instead of `train_evaluator`.
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=["loss","accuracy","roc_auc","roc_curve"],
        global_step_transform=global_step_from_engine(trainer),
    )

    # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
    # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
    # `trainer` instead of `evaluator`.
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["loss","accuracy","roc_auc","roc_curve"],
        global_step_transform=global_step_from_engine(evaluator)
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        param_name='lr'  # optional
    )

    # Attach the logger to the trainer to log model's weights norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=WeightsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsHistHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=GradsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=GradsHistHandler(model)
    )

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)
    tb_logger.close() #IMPORTANT!
    if save_path!="":
        torch.save(model.state_dict(), save_path)


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