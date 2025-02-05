import torch
import threading

class ReduceOp(torch.distributed.ReduceOp):
    def __init__(self, op):
        super().__init__(op)

num_threads = 4
use_thread_communication = False

result = None
barrier = threading.Barrier(num_threads, timeout=None)
lock = threading.Lock()

thread_mappings = {}

def init(total_num_threads):
    global num_threads
    num_threads = total_num_threads
    global barrier
    barrier = threading.Barrier(num_threads, timeout=None)
    global use_thread_communication
    use_thread_communication = True

def if_use_thread_communication():
    return use_thread_communication

def get_thread_index():
    return thread_mappings[threading.get_ident()]

def all_reduce(tensor, op=ReduceOp.SUM, group=None):
    lock.acquire()
    global result
    if op == ReduceOp.SUM:
        if result is None:
            result = tensor.clone()
        else:
            result += tensor
    elif op == ReduceOp.MAX:
        if result is None:
            result = tensor.clone()
        else:
            for i in range(result.shape[0]):
                result[i] = max(result[i], tensor[i])
    lock.release()
    barrier.wait()
    tensor.copy_(result)
    barrier.wait()
    result = None
    barrier.wait()

def _all_gather_base(tensor_list, tensor, op=ReduceOp.SUM, group=None):
    lock.acquire()
    global result
    if result is None:
        result = []
    result.append((get_thread_index(),tensor))
    lock.release()
    barrier.wait()
    for x in result:
        tensor_list[x[0]]=x[1].clone()
    barrier.wait()
    result = None
    barrier.wait()

def _reduce_scatter_base(tensor, tensor_list, op=ReduceOp.SUM, group=None):
    lock.acquire()
    global result
    if result is None:
        result = tensor_list
    else:
        for i in range(num_threads):
            result[i] += tensor_list[i]
    lock.release()
    barrier.wait()
    tensor.copy_(result[get_thread_index()])
    barrier.wait()
    result = None
    barrier.wait()

def broadcast(tensor, src, group=None):
    global result
    if get_thread_index() == src:
        result = tensor
    barrier.wait()
    tensor.copy_(result)
    barrier.wait()
    result = None
    barrier.wait()