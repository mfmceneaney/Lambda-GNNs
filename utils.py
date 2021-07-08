###############################
# Matthew McEneaney
# 7/8/21
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

# Utility Imports
import datetime, os

def load_graph_dataset(dataset="ldata_train_6_23",batch_size=256,drop_last=False,shuffle=True,num_workers=1,pin_memory=True, verbose=True):
    # Load training data
    train_dataset = LambdasDataset(dataset+"_train") # Make sure this is copied into ~/.dgl folder
    train_dataset.load()
    num_labels = train_dataset.num_labels
    node_feature_dim = train_dataset.graphs[0].ndata["data"].shape[0]
    if verbose: print("*** NODE_FEATURE_DIM *** = ",node_feature_dim)#DEBUGGING

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
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)

    return train_loader, val_loader, num_labels, node_feature_dim

def train(args, model, device, train_loader, val_loader, optimizer, scheduler, criterion, max_epochs,
            dataset="ldata_6_23_big", log_interval=10,log_dir="./logs/",save_path="./torch_models",verbose=True):

    # Make sure log/save directories exist
    try:
        os.mkdir(os.path.join(log_dir,"tb_logs/tmp"))
    except Exception:
        if verbose: print("Could not create directory:",os.path.join(log_dir,"tb_logs/tmp"))

    # Logs for matplotlib plots
    logs={'train':[], 'val':[]}

    # Create trainer
    def train_step(engine, batch):
        model.train()
        x, y   = batch
        x      = x.to(device)
        y      = y.to(device)
        y_pred = model(x,x.ndata["data"].float())
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
        x, y   = batch
        x      = x.to(device)
        y      = y.to(device)
        y_pred = model(x,x.ndata["data"].float())#TODO: Modify this so it works with PFN/EFN as well?
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

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(trainer):
    if verbose: print(f"\rEpoch[{trainer.state.epoch} : " +
          f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
          f"Loss: {trainer.state.output['loss']:.3f} Accuracy: {trainer.state.output['accuracy']:.3f}",end='')

@trainer.on(Events.EPOCH_COMPLETED)
def stepLR(trainer):
    prev_lr = scheduler.get_last_lr()
    scheduler.step()
    new_lr = scheduler.get_last_lr()
    if prev_lr != new_lr:
        if verbose: print(f"\nLR: {prev_lr[0]:.4f} -> {new_lr[0]:.4f}",end="")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    metrics = evaluator.run(train_loader).metrics
    logs['train'].append({metric:metrics[metric] for metric in metrics.keys()})
    if verbose: print(f"\nTraining Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['loss']:.4f} Avg accuracy: {metrics['accuracy']:.4f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    metrics = evaluator.run(val_loader).metrics
    logs['val'].append({metric:metrics[metric] for metric in metrics.keys()})
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

    # Create training/validation loss plot
    f = plt.figure()
    plt.title('Loss per epoch')
    plt.plot(epoch_losses,label="training")
    plt.plot(val_losses,label="validation")
    plt.legend(loc='upper right', frameon=False)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    if verbose: plt.show()
    f.savefig(os.path.join(log_dir,'training_metrics_loss_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))

    # Create training/validation accuracy plot
    f = plt.figure()
    plt.title('Accuracy per epoch')
    plt.plot(epoch_accs,label="training")
    plt.plot(val_accs,label="validation")
    plt.legend(loc='lower right', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if verbose: plt.show()
    f.savefig(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))


def evaluate(model, device, test_dataloader,dataset="ldata_6_23_big",verbose=True):

    model.eval()
    prediction = model(test_bg,test_bg.ndata["data"].float())
    test_bg    = dgl.batch(test_dataset.graphs)
    test_Y     = test_dataset.labels.clone().detach().float().view(-1, 1) #IMPORTANT: keep .view() here
    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)
    if verbose: print('Accuracy of predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

    # Get ROC curve
    pfn_fp, pfn_tp, threshs = roc_curve(test_Y.detach().numpy(), probs_Y[:,1].detach().numpy())

    # Get area under the ROC curve
    auc = roc_auc_score(test_Y.detach().numpy(), probs_Y[:,1].detach().numpy())

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
    if verbose: plt.show()
    f.savefig(model.name+"_AUC_"+datetime.datetime.now().strftime("%F")+dataset_name+"_nEps"+str(num_epochs)+".png")

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
    if verbose: plt.show()
    f.savefig(model.name+"_test_decisions_"+datetime.datetime.now().strftime("%F")+dataset_name+"_nEps"+str(num_epochs)+".png")

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