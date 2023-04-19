import torch
from os.path as osp

def parse(argfile):
    """
    Arguments
    ---------
    argfile : path

    Returns
    -------
    args   : list
    kwargs : dict
    """
    argfile = ops.expanduser(argfile)
    args, kwargs = [], {}
    return args, kwargs

def getModel(model, name, path='', device=torch.device('cpu')):
    """
    Arguments
    ---------
    model : torch.nn.module
    id    : int
    """
    args, kwargs = parse(argfile)
    m = model(*args,**kwargs) 
    m.load_state_dict(
        torch.load(
            osp.expanduser(osp.join(path,name)),
            map_location=device
        )
        )
    return m

    