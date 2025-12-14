#!/usr/bin/env python
# coding: utf-8
import torch
import torch.distributed as dist

def gather_tensor(inp, world_size=None, dist_=True, to_numpy=False):
    """Gather tensor in the distributed setting.

    Args:
        inp (torch.tensor): 
            Input torch tensor to gather.
        world_size (int, optional): 
            Dist world size. Defaults to None. If None, world_size = dist.get_world_size().
        dist_ (bool, optional):
            Whether to use all_gather method to gather all the tensors. Defaults to True.
        to_numpy (bool, optional): 
            Whether to return numpy array. Defaults to False.

    Returns:
        (torch.tensor || numpy.ndarray): Returned tensor or numpy array.
    """
    inp = torch.stack(inp)
    if dist_:
        if world_size is None:
            world_size = dist.get_world_size()
        gather_inp = [torch.ones_like(inp) for _ in range(world_size)]
        dist.all_gather(gather_inp, inp)
        gather_inp = torch.cat(gather_inp)
    else:
        gather_inp = inp

    if to_numpy:
        gather_inp = gather_inp.cpu().numpy()

    return gather_inp

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self, name='metric', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

@torch.no_grad()
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt


def update_meter(meter, value, size, is_dist=False):
    if is_dist:
        meter.update(reduce_tensor(value.data).item(), size)
    else:
        meter.update(value.item(), size)
    return meter
