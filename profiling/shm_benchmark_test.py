import torch
import time
import shm_benchmark
import multiprocessing as mp

def sender(numel, index):
    tensor = torch.ones(numel, dtype=torch.float16, device='cuda')
    start = time.time()
    shm_benchmark.write_tensor(tensor, index)
    end = time.time()
    print(f"[Sender] Sent tensor index: {index}, elapsed: {(end - start)*1000:.2f} ms")

def receiver(numel):
    time.sleep(1)
    start = time.time()
    tensor, index = shm_benchmark.read_tensor(numel)
    end = time.time()
    print(f"[Receiver] Received tensor index: {index}, elapsed: {(end - start)*1000:.2f} ms")
    return tensor, index

if __name__ == '__main__':
    tensor_mb = 10
    numel = tensor_mb * 1024 * 1024 // 2

    p_send = mp.Process(target=sender, args=(numel, 42))
    p_recv = mp.Process(target=receiver, args=(numel,))

    p_send.start()
    p_recv.start()

    p_send.join()
    p_recv.join()

    shm_benchmark.cleanup()
