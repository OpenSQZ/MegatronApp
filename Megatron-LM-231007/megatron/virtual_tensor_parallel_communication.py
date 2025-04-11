import torch
import threading
import inc.torch as dist
import queue
import time
import logging

logging.basicConfig(level=logging.DEBUG)

class ReduceOp(torch.distributed.ReduceOp):
    def __init__(self, op):
        super().__init__(op)

class Handle():
    def __init__(self):
        pass
    def wait(self):
        pass

num_threads = 4
num_forward_ranks = 2
use_thread_communication = False

result = None
new_tensor = None
thread_barrier = threading.Barrier(num_threads, timeout=None)
lock = threading.Lock()
processing_done = threading.Event()

thread_mappings = {}
requests_queue = queue.Queue()
request_args = None 
request_func = None 
request_group = None 

p2p_queue = queue.Queue()
p2p_request_args = None
p2p_reqs_list = None

global_request_group = None
global_request_args = None
global_request_func = None

listening = False

controller = None

finished_thread = None
p2p_finished_thread = None
_GLOBAL_RANK_INFO = None
_FORWARD_CONTROLLER_GROUP = None
_CONTROLLER_GROUP_RANKS = None

irecv = dist.irecv
isend = dist.isend

# class P2POp(torch.distributed.P2POp):
#     def __init__(self, op, tensor, peer=None, group=None, tag=0):
#         super().__init__(op, tensor, peer, group, tag)

def get_virtual_rank(rank):
    if _GLOBAL_RANK_INFO is None:
        return rank
    for s in range(len(_GLOBAL_RANK_INFO)):
        if _GLOBAL_RANK_INFO[s] <= 0:
            if rank == 0:
                if _GLOBAL_RANK_INFO[s] == 0:
                    return s + get_thread_index()
                else:
                    return s
            rank -= 1
    assert False, 'rank out of world_size'
    return None

def get_real_rank(rank):
    if _GLOBAL_RANK_INFO is None:
        return rank
    s = 0
    for j in range(0,rank+1):
        if _GLOBAL_RANK_INFO[j] <= 0:
            s += 1
    return s - 1

def compress(group):
    s = 0
    for i in group:
        s = s | (1<<i)
    return s

def is_ready(BitVector, group):
    state = compress(group)
    for i in group:
        if BitVector[i] != state:
            return False
    return True

def is_forward_rank(rank):
    return rank % (num_threads+1) == 0

visited = None
def DFS(graph, rank):
    if visited[rank] == 1:
        return True
    visited[rank] = 1
    for i in range(0,dist.get_world_size()):
        if (graph[rank] >> i) & 1:
            if (not ((graph[i] >> rank) & 1)) or (not DFS(graph, i)):
                return False
    return True

class Communication_Controller (threading.Thread):
    def __init__(self, _CONTROLLER_GROUP):
        threading.Thread.__init__(self)
        self.result = None
        self._CONTROLLER_GROUP = _CONTROLLER_GROUP
        self.BitVector = torch.zeros([2*num_threads+1, dist.get_world_size()], dtype = int)
        self.rank = dist.get_rank()

    def run(self):
        global listening
        global request_args
        global request_func
        global request_group
        global global_request_func
        global global_request_args
        global visited
        # print(listening)
        ts = 0
        while listening:
            # time.sleep(0.01)
            # torch.cuda.synchronize()
            # while not p2p_queue.empty():
            #     request = p2p_queue.get()
            #     print('req find', dist.get_rank())
                # dist.batch_isend_irecv(request[1])
                # request[0][0].op = None
                # for i in request[0]:
                #     print('@',i.peer, i.group)
                # dist.batch_isend_irecv(request[0])
                # print('req finished')
                # p2p_reqs_list[request[1]] = dist.batch_isend_irecv(request[0])
                # print('req finished')
            # print('running')
            # Tasks_done = True

            # for i in range(0,num_threads):
            #     if self.BitVector[i][self.rank] != 0:
            #         Tasks_done = False
            #     if p2p_finished_thread[i]:
            #         Tasks_done = False

            # while Tasks_done and requests_queue.empty() and p2p_queue.empty() and global_request_func is None:
            #     time.sleep(0.05)

            while not requests_queue.empty():
                request = requests_queue.get()
                self.BitVector[request[0]][self.rank] = compress(request[3][1])
                request_func[request[0]] = request[1]
                request_args[request[0]] = request[2]
                request_group[request[0]] = request[3]

            while not p2p_queue.empty():
                request = p2p_queue.get()
                for i in range(len(request[1])):
                    if is_forward_rank(request[1][i].peer):
                        self.BitVector[num_threads+1+request[0]][self.rank] += (1 << request[1][i].peer)
                    # request[1][i].peer = get_real_rank(request[1][i].peer)
                p2p_request_args[request[0]] = request[1]
                p2p_finished_thread[request[0]] = 1

            tmp_BitVector = self.BitVector.clone()
            if global_request_func is not None:
                tmp_BitVector[num_threads][self.rank] = compress(global_request_group[1])
            # print('before:',self.BitVector)
            dist.all_reduce(tmp_BitVector, ReduceOp.SUM, self._CONTROLLER_GROUP)
            # ts += 1
            # if ts % 10 ==0:
            #     print('controller running, state:')
            #     print(tmp_BitVector)
            if tmp_BitVector[num_threads][self.rank] != 0:
                if is_ready(tmp_BitVector[num_threads],global_request_group[1]):
                    # print(global_request_func, global_request_args)
                    # print('dealing with global', global_request_func, tmp_BitVector, global_request_group)
                    global_request_func(*global_request_args,global_request_group[0])
                    # print(global_request_func, 'finished')
                    global_request_func = None
                # print('after:',tmp_BitVector)
            for i in range(num_threads):
                if self.BitVector[i][self.rank] != 0 and is_ready(tmp_BitVector[i],request_group[i][1]):
                    # print('dealing with', i, tmp_BitVector, request_group[i][1])
                    request_func[i](*request_args[i], request_group[i][0])
                    # print('finished', i)
                    self.BitVector[i][self.rank] = 0
                    finished_thread[i] = 0
            
            for i in range(num_threads+1,num_threads*2+1):
                index = i - num_threads - 1
                if p2p_finished_thread[index]:
                    visited = [0 for i in range(0, dist.get_world_size())]
                    if DFS(tmp_BitVector[i], self.rank):
                        # print('dealing with', i, tmp_BitVector)
                        self.BitVector[i][self.rank] = 0
                        p2p_finished_thread[index] = 0
                        # print('!!!!!!!!!!!!!!!!!!!!!!!!')
                        p2p_reqs_list[index] = dist.batch_isend_irecv(p2p_request_args[index])
                        p2p_request_args[index] = None
                        # print('finished', i)
            # if tmp_BitVector[num_threads][0] == num_forward_ranks:
            #     print('?')
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

def set_virtual_rank_info(rank_info):
    global _GLOBAL_RANK_INFO
    _GLOBAL_RANK_INFO = rank_info

def init(total_num_threads, total_num_forward_ranks, _CONTROLLER_GROUP, group_ranks):
    global num_threads
    num_threads = total_num_threads
    global num_forward_ranks
    num_forward_ranks = total_num_forward_ranks
    global thread_barrier
    thread_barrier = threading.Barrier(num_threads, timeout=None)
    global use_thread_communication
    use_thread_communication = True
    global listening
    listening = True
    global request_args
    global request_func
    global request_group
    global p2p_reqs_list
    global p2p_request_args
    request_args = [None for i in range(0, num_threads)]
    request_func = [None for i in range(0, num_threads)]
    request_group = [None for i in range(0, num_threads)]
    p2p_reqs_list = [None for i in range(0, num_threads)]
    p2p_request_args = [None for i in range(0, num_threads)]
    global global_request_func
    global global_request_args
    global_request_args = None
    global_request_func = None
    global finished_thread
    global p2p_finished_thread
    finished_thread = [0 for i in range(0, num_threads)]
    p2p_finished_thread = [0 for i in range(0, num_threads)]
    global _FORWARD_CONTROLLER_GROUP
    global _CONTROLLER_GROUP_RANKS
    _FORWARD_CONTROLLER_GROUP = _CONTROLLER_GROUP
    _CONTROLLER_GROUP_RANKS = group_ranks
    # print('initiallize:',group_ranks)

def start_controller():
    global controller
    # global listening
    # listening = True
    controller = Communication_Controller(_FORWARD_CONTROLLER_GROUP)
    controller.start()

def stop_controller():
    global listening
    listening = False
    while controller.is_alive():
        pass

def if_use_thread_communication():
    return use_thread_communication

def set_use_thread_communication():
    use_thread_communication = True

def get_thread_index():
    global thread_mappings
    # print('#', threading.get_ident())
    if threading.get_ident() in thread_mappings:
        return thread_mappings[threading.get_ident()]
    else:
        return -1

def is_initialized():
    return dist.is_initialized()

def P2POp(op, tensor, peer=None, group=None, tag=0):
    if peer is not None:
        peer = get_real_rank(peer)
    return dist.P2POp(op, tensor, peer, group, tag)

def barrier(group = None):
    if not use_thread_communication:
        dist.barrier()
    else:
        if get_thread_index() == 0:
            global global_request_func
            global global_request_args
            global global_request_group
            if group is None:
                global_request_group = [None, _CONTROLLER_GROUP_RANKS]
            else:
                global_request_group = group
            global_request_args = ()
            global_request_func = dist.barrier
        # global controller
        # print(dist.get_rank(),controller.is_alive())
        thread_barrier.wait()
        while global_request_func is not None:
            pass

def batch_isend_irecv(p2p_op_list):
    # lock.acquire()
    # global listening
    # tmp = listening
    # if tmp:
    #     stop_controller()
    # print('stopped',dist.get_rank())
    if not use_thread_communication:
        return dist.batch_isend_irecv(p2p_op_list)
    
    # print('batch_isend_irecv start',dist.get_rank())
    index = get_thread_index()
    p2p_queue.put([index, p2p_op_list])
    while p2p_reqs_list[index] is None:
        pass
    reqs = p2p_reqs_list[index]
    p2p_reqs_list[index] = None
    # reqs = dist.batch_isend_irecv(p2p_op_list)
    # if tmp:
    #     start_controller()
    # print('batch_isend_irecv finished',dist.get_rank())
    # lock.release()
    return reqs

def get_world_size(group = None,):
    if isinstance(group, list):
        return dist.get_world_size(group[0])
    elif isinstance(group, int):
        return num_threads
    elif group is None:
        if _GLOBAL_RANK_INFO is None:
            return dist.get_world_size()
        else:
            return len(_GLOBAL_RANK_INFO)
    else:
        return dist.get_world_size(group)

def get_rank(group = None,):
    if isinstance(group, list):
        return dist.get_rank(group[0])
    elif isinstance(group, int):
        return get_thread_index()
    elif group is None:
        return get_virtual_rank(dist.get_rank())
    else:
        return dist.get_world_size(group)

def init_process_group(backend, world_size, rank, timeout,):
    dist.init_process_group(backend = backend, world_size = world_size, rank = rank, timeout = timeout)

def set_thread_index(rank):
    global thread_mappings
    # print('%', threading.get_ident(), rank)
    thread_mappings[threading.get_ident()] = rank

def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    global use_thread_communication
    global result
    # print('start_allreduce', tensor, op, group)
    # print('all_reduce start')
    if not use_thread_communication:
        # print('all_reduce start')
        start_time = time.time()
        handle = dist.all_reduce(tensor, op, group, async_op)
        end_time = time.time()
        print(dist.get_rank(),"all_reduce ~ duration",end_time-start_time)
        # print('all_reduce finished')
        return handle
    elif isinstance(group, int):
        thread_barrier.wait()
        start_time = time.time()
        lock.acquire()
        if op == ReduceOp.SUM:
            if result is None:
                result = tensor.clone()
            else:
                result.add_(tensor)
        elif op == ReduceOp.MAX:
            if result is None:
                result = tensor.clone()
            else:
                result = torch.max(result, tensor)
        lock.release()
        thread_barrier.wait()
        tensor.copy_(result)
        thread_barrier.wait()
        result = None
        thread_barrier.wait()
        end_time = time.time()
        print(dist.get_rank(),"all_reduce ! duration",end_time-start_time)
    elif group is None or len(group) == 3:
        start_time = time.time()
        global global_request_func
        global global_request_args
        global global_request_group
        if group is None:
            group = [None, _CONTROLLER_GROUP_RANKS]
        if get_thread_index() == -1:
            global_request_group = group
            global_request_args = (tensor, op)
            global_request_func = dist.all_reduce
            while global_request_func is not None:
                pass
        else:
            # print('#############')
            lock.acquire()
            print(tensor)
            if op == ReduceOp.SUM:
                if result is None:
                    result = tensor.clone()
                else:
                    result += tensor
            elif op == ReduceOp.MAX:
                if result is None:
                    result = tensor.clone()
                else:
                    result = torch.max(result, tensor)
            elif op == ReduceOp.MIN:
                if result is None:
                    result = tensor.clone()
                else:
                    result = torch.min(result, tensor)
            lock.release()
            thread_barrier.wait()
            if get_thread_index() == 0:
                global_request_group = group
                global_request_args = (result, op)
                global_request_func = dist.all_reduce
            thread_barrier.wait()
            # print(result,get_thread_index())
            while global_request_func is not None:
                # print(global_request_func)
                pass
            # print(result)
            tensor.copy_(result)
            thread_barrier.wait()
            result = None
            thread_barrier.wait()
        end_time = time.time()
        print(dist.get_rank(),"all_reduce @ duration",end_time-start_time)
    else:
        start_time = time.time()
        index = get_thread_index()
        # print('@',index)
        # print(finished_thread)
        finished_thread[index] = 1
        # print(group)
        requests_queue.put((index, dist.all_reduce, (tensor, op), group))
        while controller.get_status(index):
            pass
        end_time = time.time()
        print(dist.get_rank(),"all_reduce # duration",end_time-start_time)
        # print(group)
    # print('all_reduce finished')
    if async_op:
        return Handle()

def get_global_shape(tensor):
    tensor_list = []
    for i in _GLOBAL_RANK_INFO:
        if i == 0:
            tensor_list.append(torch.zeros((num_threads,) + tuple(tensor.shape), dtype = tensor.dtype, device = torch.cuda.current_device()))
        elif i == -1:
            tensor_list.append(torch.zeros(tensor.shape, dtype = tensor.dtype, device = torch.cuda.current_device()))
    return tensor_list

def _all_gather_base(tensor_list, tensor, group=None, async_op=False):
    global use_thread_communication
    global new_tensor
    global result
    # print('all_gather start')
    if not use_thread_communication:
        # print('all_gather start')
        if group is None:
            result = get_global_shape(tensor)
            print("all_gather", dist.get_rank(), result)
            handle = dist.all_gather(result, tensor, group, async_op)
            print("all_gather", dist.get_rank(), "finished")
            tensor_list.copy_(result.view(-1))
        else:
            start_time = time.time()
            handle = dist._all_gather_base(tensor_list, tensor, group, async_op)
            end_time = time.time()
            print(dist.get_rank(),"_all_gather_base ~ duration",end_time-start_time)
        # print('all_gather finished')
        return handle
    elif isinstance(group, int):
        start_time = time.time()
        lock.acquire()
        if result is None:
            result = []
        result.append((get_thread_index(),tensor))
        lock.release()
        thread_barrier.wait()
        for x in result:
            tensor_list[x[0]]=x[1].clone()
        thread_barrier.wait()
        result = None
        thread_barrier.wait()
        end_time = time.time()
        print(dist.get_rank(),"_all_gather_base @ duration",end_time-start_time)
    elif group is None:
        global global_request_func
        global global_request_args
        global global_request_group
        lock.acquire()
        if new_tensor is None:
            new_tensor = torch.zeros((num_threads,) + tuple(tensor.shape), dtype = tensor.dtype, device = torch.cuda.current_device())
            result = get_global_shape(tensor)
        new_tensor[get_thread_index()] = tensor.clone()
        lock.release()
        thread_barrier.wait()
        if get_thread_index() == 0:
            global_request_group = [None, _CONTROLLER_GROUP_RANKS]
            global_request_args = (result, new_tensor)
            global_request_func = dist.all_gather
        thread_barrier.wait()
        while global_request_func is not None:
            pass
        tensor_list.copy_(result.view(-1))
        thread_barrier.wait()
        result = None
        new_tensor = None
        thread_barrier.wait()
    else:
        index = get_thread_index()
        finished_thread[index] = 1
        requests_queue.put((index, dist._all_gather_base, (tensor_list, tensor), group))
        # print(group)
        while controller.get_status(index):
            pass
    # print('all_gather finished')
    if async_op:
        return Handle()

def _reduce_scatter_base(tensor, tensor_list, op=ReduceOp.SUM, group=None, async_op=False):
    global use_thread_communication
    if not use_thread_communication:
        return dist._reduce_scatter_base(tensor, tensor_list, op, group, async_op)
    elif isinstance(group, int):
        start_time = time.time()
        lock.acquire()
        global result
        if result is None:
            result = tensor_list
        else:
            for i in range(num_threads):
                result[i].add_(tensor_list[i])
        lock.release()
        thread_barrier.wait()
        tensor.copy_(result[get_thread_index()])
        thread_barrier.wait()
        result = None
        thread_barrier.wait()
        end_time = time.time()
        print(dist.get_rank(),"_reduce_scatter_base @ duration",end_time-start_time)
    else:
        start_time = time.time()
        index = get_thread_index()
        requests_queue.put((index, dist._reduce_scatter_base, (tensor, tensor_list, op), group))
        # print(group)
        while controller.get_status(index):
            pass
        end_time = time.time()
        print(dist.get_rank(),"_reduce_scatter_base # duration",end_time-start_time)
    if async_op:
        return Handle()

def broadcast(tensor, src, group=None):
    global use_thread_communication
    # print('$', src, group)
    if not use_thread_communication:
        src = get_real_rank(src)
        # print('broadcast start')
        dist.broadcast(tensor, src, group)
        # print('broadcast finished')
    elif group is None:
        src = get_real_rank(src)
        global global_request_func
        global global_request_args
        global global_request_group
        global_request_group = (None, _CONTROLLER_GROUP_RANKS)
        global_request_args = (tensor, src, group)
        global_request_func = dist.broadcast
        while global_request_func is not None:
            pass
    elif isinstance(group, int):
        global result
        # print('@', src, _GLOBAL_RANK_INFO)
        start_time = time.time()
        if get_thread_index() == _GLOBAL_RANK_INFO[src]:
            result = tensor
        thread_barrier.wait()
        tensor.copy_(result)
        thread_barrier.wait()
        result = None
        thread_barrier.wait()
        end_time = time.time()
        print(dist.get_rank(),"broadcast # duration",end_time-start_time)
    else:
        src = get_real_rank(src)
        index = get_thread_index()
        finished_thread[index] = 1
        # print(group)
        requests_queue.put((get_thread_index(), dist.broadcast, (tensor, src), group))
        while controller.get_status(index):
            pass