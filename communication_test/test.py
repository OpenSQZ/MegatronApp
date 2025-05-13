# import inc.torch as dist
#import torch.distributed as dist
import inc.torch as dist

import threading
import torch
import time

threadpool = []

torch.distributed.init_process_group(backend='nccl')

torch.cuda.set_device(dist.get_rank())

ranks = [0, 1]

group = torch.distributed.new_group(ranks)

# dist.init(4, 2, group)

# X = torch.tensor([2.0])

# torch.distributed.all_reduce(X, group = group)

# print(X)

# class Threadtest(threading.Thread):
#     def __init__(self, tensor, tensor_list, rank):
#         threading.Thread.__init__(self)
#         self.tensor=tensor
#         self.rank=rank
#         self.tensor_list=tensor_list
#         self.result=torch.tensor([0.0])
#         self.result2=[torch.tensor([0.0]),torch.tensor([0.0])]
#     def run(self):
#         dist.set_thread_index(self.rank)
#         # print(dist.get_thread_index())
#         dist._all_gather_base(self.result2, self.tensor, group = group)
#         dist.all_reduce(self.tensor, op=dist.ReduceOp.SUM, group = group)
#         # dist._reduce_scatter_base(self.result, self.tensor_list)

# for i in range(4):
#     tensor = torch.tensor([i*1.0])
#     tensor_list = [torch.tensor([i*1.0]), torch.tensor([i*2.0]), torch.tensor([i*3.0]), torch.tensor([i*4.0])]
#     threadpool.append(Threadtest(tensor, tensor_list, i))

# for t in threadpool:
#     t.start()

# for t in threadpool:
#     t.join()

# if torch.distributed.get_rank() == 0:
#     for t in threadpool:
#         print(t.tensor, t.result, t.result2)

class Thread1(threading.Thread):
    def __init__(self, ):
        threading.Thread.__init__(self)
    def run(self):
        import time
        time.sleep(1)
        dist.barrier()
        # dist._reduce_scatter_base(self.result, self.tensor_list)

class Thread2(threading.Thread):
    def __init__(self, ):
        threading.Thread.__init__(self)
    def run(self):
        self.tensor=torch.tensor([1.0])
        dist.all_reduce(self.tensor, dist.ReduceOp.SUM, group)
        print(self.tensor)

# if dist.get_rank() == 0:
#     t1 = Thread1()
#     t2 = Thread2()
#     t1.start()
#     t2.start()
#     t1.join()
#     t2.join()
# else:
#     t1 = Thread1()
#     t2 = Thread2()
#     t2.start()
#     # import time
#     # time.sleep(1)
#     t1.start()
#     t1.join()
#     t2.join()

def cp(x, y):
    x.copy_(y.view(-1))

tensor=torch.tensor([3.0, 4.0], device = 'cuda')
tensorp=torch.tensor([1.0], device = 'cuda')
tensor_list = torch.zeros(2 * (tensor.shape),
                            dtype=torch.float,
                            device=torch.cuda.current_device())
x = tensor_list.clone()
dist._all_gather_base(x, tensor.view(-1), group)
cp(tensor_list.view(-1), x)
print(tensor_list)
# if dist.get_rank() == 1:
#     dist.all_reduce(tensor, dist.ReduceOp.SUM, group)
# time.sleep(1)
# if dist.get_rank() in ranks:
#     print(dist.get_rank())
#     dist.all_reduce(tensor, dist.ReduceOp.SUM, group)
#     print(tensor)
# if dist.get_rank() == 2:
#     op = dist.P2POp(dist.isend, tensor, 0, group)
#     op_list = [op]
#     reqs = dist.batch_isend_irecv(op_list)
# time.sleep(1)
# if dist.get_rank() == 0:
#     op = dist.P2POp(dist.isend, tensor, 1, group)
#     op_list = [op]
#     reqs = dist.batch_isend_irecv(op_list)
#     print('sent')
#     reqs[0].wait()
#     # print('sent')
#     # reqs[0].wait()
#     # print('done')
# if dist.get_rank() == 1:
#     # while True:
#     #     pass
#     op = dist.P2POp(dist.irecv, tensor, 0, group)
#     op_list = [op]
#     # op = dist.P2POp(dist.isend, tensorp, 2, group)
#     # op_list.append(op)
#     reqs = dist.batch_isend_irecv(op_list)
#     # reqs[0].wait()
#     # tensor[0] += 1.0
#     # op = dist.P2POp(dist.isend, tensor, 0, group)
#     # op_list = [op]
#     # reqs = dist.batch_isend_irecv(op_list)
#     # print('sent')
#     # reqs[0].wait()
# time.sleep(1)
# if dist.get_rank() == 0:
#     op = dist.P2POp(dist.irecv, tensorp, 2, group)
#     op_list = [op]
#     reqs = dist.batch_isend_irecv(op_list)
#     print('recv')
#     reqs[0].wait()
#     print(tensorp)