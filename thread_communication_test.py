import virtual_tensor_parallel_communication as dist
#import torch.distributed as dist

import threading
import torch

class Threadtest(threading.Thread):
    def __init__(self, tensor, tensor_list, rank):
        threading.Thread.__init__(self)
        self.tensor=tensor
        self.rank=rank
        self.tensor_list=tensor_list
        self.result=torch.tensor([0.0])
        self.result2=[torch.tensor([0.0]),torch.tensor([0.0]),torch.tensor([0.0]),torch.tensor([0.0])]
    def run(self):
        dist.thread_mappings[threading.get_ident()]=self.rank
        dist._all_gather_base(self.result2, self.tensor)
        dist.all_reduce(self.tensor, op=dist.ReduceOp.MAX)
        dist._reduce_scatter_base(self.result, self.tensor_list)

threadpool = []

for i in range(4):
    tensor = torch.tensor([i*1.0])
    tensor_list = [torch.tensor([i*1.0]), torch.tensor([i*2.0]), torch.tensor([i*3.0]), torch.tensor([i*4.0])]
    threadpool.append(Threadtest(tensor, tensor_list, i))

for t in threadpool:
    t.start()

for t in threadpool:
    t.join()

for t in threadpool:
    print(t.tensor, t.result, t.result2)