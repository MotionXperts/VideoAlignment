import torch.distributed as dist
from icecream import ic


def is_root_proc():
    if dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True

def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def all_reduce(tensors,average=True):
    if not dist.is_available():
        return tensors
    if not dist.is_initialized():
        return tensors
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensors
    ic(tensors)
    for tensor in tensors:
        ic(tensor)
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors