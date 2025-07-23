import torch
import time
import shm_tensor_new_rdma_pre_alloc
import shm_tensor_new_rdma_multithread
import shm_tensor_new_rdma
import multiprocessing as mp
import os

def sender(numel, iters):
    shm_tensor_new_rdma.init_shared_memory(numel, 0, iters, 1, 2, [])
    shm_tensor_new_rdma.thread_pool(1, 2, iters, False, False, 1, 1, 0, iters, 0, 0, 0, numel, True)
    for i in range(iters):
        tensor = torch.randn(numel, dtype=torch.float16, device='cuda')
        print(f"send start at {time.time()}, {i}")
        torch.cuda.synchronize()
        shm_tensor_new_rdma.put_forward_tensor(i, tensor)
        torch.cuda.synchronize()
        print(f"send end at {time.time()}, {i}")
    time.sleep(1)

def receiver(numel, iters):
    shm_tensor_new_rdma.init_shared_memory(numel, 1, iters, 1, 2, [])
    shm_tensor_new_rdma.thread_pool(1, 2, iters, False, False, 1, 1, 0, 0, iters, 0, 0, numel, True)
    for i in range(iters):
        print(f"receive start at {time.time()}, {i}")
        torch.cuda.synchronize()
        while shm_tensor_new_rdma.get_forward_tensor(i) is None:
            pass
        torch.cuda.synchronize()
        print(f"receive end at {time.time()}, {i}")

if __name__ == '__main__':
    numel = 2048 * 2048 * 8
    os.system("rm -rf /dev/shm/sem.*")
    os.system("rm -rf /dev/shm/forward_*")
    os.system("rm -rf /dev/shm/backward_*")
    iters = 32
    p_send = mp.Process(target=sender, args=(numel, iters))
    p_recv = mp.Process(target=receiver, args=(numel, iters))

    p_send.start()
    p_recv.start()

    p_send.join()
    p_recv.join()
    os.system("rm -rf /dev/shm/sem.*")
    os.system("rm -rf /dev/shm/forward_*")
    os.system("rm -rf /dev/shm/backward_*")
