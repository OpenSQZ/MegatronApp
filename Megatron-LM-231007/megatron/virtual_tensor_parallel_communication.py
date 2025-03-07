import torch
import threading
import inc.torch as dist
import queue
import time

class ReduceOp(torch.distributed.ReduceOp):
    def __init__(self, op):
        super().__init__(op)

num_threads = 4
num_forward_ranks = 2
use_thread_communication = False

result = None
barrier = threading.Barrier(num_threads, timeout=None)
lock = threading.Lock()
processing_done = threading.Event()

thread_mappings = {}
requests_queue = queue.Queue()
request_args = None 
request_func = None 

listening = False

controller = None

finished_thread = None

class Communication_Controller (threading.Thread):
    def __init__(self, _CONTROLLER_GROUP, _CONTROLLER_GROUP_GLOO):
        threading.Thread.__init__(self)
        self.result = None
        self.BitVector = torch.zeros(num_threads, dtype = int)
        self._CONTROLLER_GROUP = _CONTROLLER_GROUP
        self._CONTROLLER_GROUP_GLOO = _CONTROLLER_GROUP_GLOO

    def run(self):
        global listening
        global request_args
        global request_func
        # print(listening)
        while listening:
            time.sleep(0.1)
            while not requests_queue.empty():
                request = requests_queue.get()
                self.BitVector[request[0]] = 1
                request_func[request[0]] = request[1]
                request_args[request[0]] = request[2]
            dist.all_reduce(self.BitVector, ReduceOp.SUM, self._CONTROLLER_GROUP)
            for i in range(num_threads):
                if self.BitVector[i] == num_forward_ranks:
                    request_func[i](*request_args[i])
                    self.BitVector[i] = 0
                    finished_thread[i] = 0
            # if requests_queue.qsize() == num_thread:
            #     if request_func is dist.all_reduce:
            #         while not requests_queue.empty():
            #             request = requests_queue.get()
            #             requests_list[request[3]] = (request[0], request[1], request[2])
            #         for i in range(0, num_thread):
            #             request_func(requests_list[i][0], op = requests_list[i][1], group = requests_list[i][2])
            #     elif request_func is dist._reduce_scatter_base:
            #         while not requests_queue.empty():
            #             request = requests_queue.get()
            #             requests_list[request[4]] = (request[0], request[1], request[2], request[3])
            #         for i in range(0, num_thread):
            #             request_func(requests_list[i][0], requests_list[i][1], op = requests_list[i][2], group = requests_list[i][3])
            #     processing_done.set()
    def get_status(self, index):
        return finished_thread[index]

def init(total_num_threads, total_num_forward_ranks, _CONTROLLER_GROUP):
    global num_threads
    num_threads = total_num_threads
    global barrier
    barrier = threading.Barrier(num_threads, timeout=None)
    global use_thread_communication
    use_thread_communication = True
    global num_forward_ranks
    num_forward_ranks = total_num_forward_ranks
    global listening
    listening = True
    global request_args
    global request_func
    request_args = [None for i in range(0, num_threads)]
    request_func = [None for i in range(0, num_threads)]
    global finished_thread
    finished_thread = [0 for i in range(0, num_threads)]
    global controller
    controller = Communication_Controller(_CONTROLLER_GROUP, None)
    controller.start()

def if_use_thread_communication():
    return use_thread_communication

def get_thread_index():
    global thread_mappings
    # print('#', threading.get_ident())
    return thread_mappings[threading.get_ident()]

def set_thread_index(rank):
    global thread_mappings
    # print('%', threading.get_ident(), rank)
    thread_mappings[threading.get_ident()] = rank

def all_reduce(tensor, op=ReduceOp.SUM, group=None, use_inter_thread_comm=False):
    if use_inter_thread_comm:
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
        dist.all_reduce(result, op = op, group = group)
        tensor.copy_(result)
        barrier.wait()
        result = None
        barrier.wait()
    else:
        # print('&', threading.get_ident())
        index = get_thread_index()
        # print('@',index)
        # print(finished_thread)
        finished_thread[index] = 1
        requests_queue.put((index, dist.all_reduce, (tensor, op, group)))
        while controller.get_status(index):
            pass

def _all_gather_base(tensor_list, tensor, group=None, use_inter_thread_comm=False):
    if use_inter_thread_comm:
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
        dist._all_gather_base(tensor_list, tensor_list, op = op, group = group)
        result = None
        barrier.wait()
    else:
        index = get_thread_index()
        finished_thread[index] = 1
        requests_queue.put((index, dist._all_gather_base, (tensor_list, tensor, group)))
        while controller.get_status(index):
            pass

def _reduce_scatter_base(tensor, tensor_list, op=ReduceOp.SUM, group=None, use_inter_thread_comm=False):
    if use_inter_thread_comm:
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
    else:
        index = get_thread_index()
        requests_queue.put((index, dist._reduce_scatter_base, (tensor, tensor_list, op, group)))
        while controller.get_status(index):
            pass

def broadcast(tensor, src, group=None, use_inter_thread_comm=False):
    if use_inter_thread_comm:
        global result
        if get_thread_index() == src:
            result = tensor
        barrier.wait()
        tensor.copy_(result)
        barrier.wait()
        result = None
        barrier.wait()
    else:
        index = get_thread_index()
        finished_thread[index] = 1
        requests_queue.put((get_thread_index(), dist.broadcast, (tensor, src, group)))
        while controller.get_status(index):
            pass