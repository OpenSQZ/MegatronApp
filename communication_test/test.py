import virtual_tensor_parallel_communication as dist
#import torch.distributed as dist

import threading
import torch

threadpool = []

torch.distributed.init_process_group(backend='gloo')

ranks = [0, 1]

group = torch.distributed.new_group(ranks)

dist.init(4, 2, group)

# X = torch.tensor([2.0])

# torch.distributed.all_reduce(X, group = group)

# print(X)

class Threadtest(threading.Thread):
    def __init__(self, tensor, tensor_list, rank):
        threading.Thread.__init__(self)
        self.tensor=tensor
        self.rank=rank
        self.tensor_list=tensor_list
        self.result=torch.tensor([0.0])
        self.result2=[torch.tensor([0.0]),torch.tensor([0.0])]
    def run(self):
        dist.set_thread_index(self.rank)
        # print(dist.get_thread_index())
        dist._all_gather_base(self.result2, self.tensor, group = group)
        dist.all_reduce(self.tensor, op=dist.ReduceOp.SUM, group = group)
        # dist._reduce_scatter_base(self.result, self.tensor_list)

for i in range(4):
    tensor = torch.tensor([i*1.0])
    tensor_list = [torch.tensor([i*1.0]), torch.tensor([i*2.0]), torch.tensor([i*3.0]), torch.tensor([i*4.0])]
    threadpool.append(Threadtest(tensor, tensor_list, i))

for t in threadpool:
    t.start()

for t in threadpool:
    t.join()

if torch.distributed.get_rank() == 0:
    for t in threadpool:
        print(t.tensor, t.result, t.result2)