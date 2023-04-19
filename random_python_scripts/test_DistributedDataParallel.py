import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch import optim
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef

class Model(nn.Module):
    """
        Simple 3 layer MLP
    """
    def __init__(self, input_size=256, hdim=256, num_classes=2):
        super(Model, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, hdim),
            nn.Sigmoid(),
            nn.Linear(hdim, hdim),
            nn.Sigmoid(),
            nn.Linear(hdim, num_classes),
            nn.Sigmoid(),
        )

        self.name = "Model"
    
    def forward(self, h):
        y = self.layer(h)
        return y
    
    # @property
    def name(self):
        """
        Name of model.
        """
        return self.name


if __name__=="__main__":

    # Initialize distributed processes
    torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')

    t1 = torch.rand((3, 3), requires_grad=True)
    t2 = torch.rand((3, 3), requires_grad=True)
    rref = rpc.remote("worker1", torch.add, args=(t1, t2))
    ddp_model = DDP(my_model)

    # Setup optimizer
    optimizer_params = [rref]
    for param in ddp_model.parameters():
        optimizer_params.append(RRef(param))

    dist_optim = DistributedOptimizer(
        optim.SGD,
        optimizer_params,
        lr=0.05,
    )

    with dist_autograd.context() as context_id:
        pred = ddp_model(rref.to_here())
        loss = loss_func(pred, target)
        dist_autograd.backward(context_id, [loss])
        dist_optim.step(context_id)

