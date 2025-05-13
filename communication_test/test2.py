import torch
import threading

def clone_tensor():
    t = torch.rand(1000, 1000, device="cuda")
    t_clone = []
    for i in range(0,10):
        t_clone.append(t.clone() -  torch.rand(1000, 1000, device="cuda"))# Uses default stream
    print("Cloning done in thread", threading.current_thread().name, t_clone[0])

# Create multiple threads running tensor.clone()
threads = [threading.Thread(target=clone_tensor) for _ in range(2)]

for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
