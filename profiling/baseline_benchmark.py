import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

def run(rank, world_size, tensor_mb=10, use_cuda=False, num_repeat=100):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    backend = 'nccl' if use_cuda else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device('cuda', rank) if use_cuda else torch.device('cpu')
    tensor_size = 2048 * 2048 * 8
    tensor = torch.randn(tensor_size, dtype=torch.float16, device=device)

    dist.barrier()

    if rank == 0:
        for i in range(num_repeat):
            print(f"send start at {time.time()}, {i}")
            torch.cuda.synchronize()
            dist.send(tensor=tensor, dst=1)
            torch.cuda.synchronize()
            print(f"send end at {time.time()}, {i}")
    elif rank == 1:
        recv_tensor = torch.empty_like(tensor)
        for i in range(num_repeat):
            print(f"receive start at {time.time()}, {i}")
            torch.cuda.synchronize()
            dist.recv(tensor=recv_tensor, src=0)
            torch.cuda.synchronize()
            print(f"receive end at {time.time()}, {i}")

    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run,
             args=(2, 10, True, 32),
             nprocs=2,
             join=True)