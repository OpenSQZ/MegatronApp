import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

def run(rank, world_size, tensor_mb=10, use_cuda=False):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    backend = 'nccl' if use_cuda else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device('cuda', rank) if use_cuda else torch.device('cpu')
    tensor_size = tensor_mb * 1024 * 1024 // 2  # float16: 2 bytes
    tensor = torch.ones(tensor_size, dtype=torch.float16, device=device)

    dist.barrier()  # 确保两个进程都准备好

    if rank == 0:
        # sender
        start = time.perf_counter()
        dist.send(tensor=tensor, dst=1)
        end = time.perf_counter()
        print(f"[Sender] Sent {tensor_mb}MB in {end - start:.6f} seconds.")
    elif rank == 1:
        # receiver
        recv_tensor = torch.empty_like(tensor)
        start = time.perf_counter()
        dist.recv(tensor=recv_tensor, src=0)
        end = time.perf_counter()
        print(f"[Receiver] Received {tensor_mb}MB in {end - start:.6f} seconds.")

    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run,
             args=(2, 10, True),  # world_size=2, tensor_mb=10MB, use_cuda=False
             nprocs=2,
             join=True)
