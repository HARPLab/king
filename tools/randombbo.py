from torch.optim.optimizer import Optimizer
import torch

class RandBBO(Optimizer):
    def __init__(self, params, distr="norm", var=None):
        if(var == None):
            var = [1 for _ in range(len(params))]
        defaults = dict(var=var, distr=distr)
        super(RandBBO, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RandBBO, self).__setstate__(state)
    
    def step(self, closure = None):
        for group in self.param_groups:
            distr = group['distr']
            var = group['var']

            for i,p in enumerate(group['params']):
                #deltas = torch.normal(mean=0.0, std=var[i]**0.5, size=p.size(), device='cuda')
                deltas = (0.4 * torch.rand(p.size(), device='cuda')) - 0.2
                p.data.add_(deltas)
