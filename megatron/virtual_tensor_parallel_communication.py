# Copyright 2025 Suanzhi Future Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import torch
import threading
import torch.distributed as dist
import queue
import time
import logging

logging.basicConfig(level=logging.DEBUG)

# Default Ops:
new_group = dist.new_group
is_available = dist.is_available
gather = dist.gather
get_global_rank = dist.get_global_rank
get_process_group_ranks = dist.get_process_group_ranks
all_gather_into_tensor = dist.all_gather_into_tensor
reduce_scatter_tensor = dist.reduce_scatter_tensor
broadcast_object_list = dist.broadcast_object_list

dummy = torch.zeros(1)

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
normal_communication = True

result = None
new_tensor = None
thread_barrier = None
lock = threading.Lock()
processing_done = threading.Event()

thread_mappings = {}
requests_queue = queue.Queue()
request_args = None 
request_func = None 
request_group = None 
request_conditions = None

p2p_queue = queue.Queue()
p2p_request_args = None
p2p_reqs_list = None
p2p_conditions = None

global_request_group = None
global_request_args = None
global_request_func = None
global_condition = None

listening = False

controller = None

finished_thread = None
p2p_finished_thread = None
_GLOBAL_RANK_INFO = None
_FORWARD_CONTROLLER_GROUP = None
_CONTROLLER_GROUP_RANKS = None

irecv = dist.irecv
isend = dist.isend

donothing = False
_GLOBAL_GROUP = None
_GLOBAL_RANKS = None
_GLOBAL_GROUP_GLOO = None

tensor_parallel_rank = 0

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
                    if tensor_parallel_rank == -1:
                        return s + max(0, get_thread_index())
                    else:
                        return s + tensor_parallel_rank
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
    return s

def backward_rank_only(ranks):
    for r in ranks:
        if _GLOBAL_RANK_INFO[r] >= 0:
            return False
    return True

def compress(group):
    s = 0
    for i in group:
        s = s | (1<<i)
    return s

def is_ready_backward(base_id, BitVector, group):
    state = compress(group)
    for i in group:
        real_i = get_virtual_rank(i - 1)
        offset = 0
        if _GLOBAL_RANK_INFO[real_i] < 0:
            offset = -_GLOBAL_RANK_INFO[real_i] - 1
        else:
            offset = base_id
        # print(i, offset, real_i)
        if BitVector[offset][i] != state:
            return False
    return True

def is_forward_rank(rank):
    return rank % (num_threads+1) == 0

visited = None
graph = None
def DFS(rank, world_size):
    if visited[rank] == 1:
        return True
    visited[rank] = 1
    for i in range(0, world_size):
        if (graph[rank] >> i) & 1:
            if (not ((graph[i] >> rank) & 1)) or (not DFS(i, world_size)):
                return False
    return True

class Backward_Controller (threading.Thread):
    def __init__(self, _CONTROLLER_GROUP, tensor_parallel_rank):
        threading.Thread.__init__(self)
        self.result = None
        self._CONTROLLER_GROUP = _CONTROLLER_GROUP
        self.BitVector = torch.zeros([2*num_threads+1, dist.get_world_size()], dtype = int)
        self.rank = dist.get_rank()
        self.tensor_parallel_rank = tensor_parallel_rank

    def run(self):
        global listening
        global visited
        global graph
        global global_request_func
        global global_request_args
        global global_request_group
        # print(listening)
        # torch.device("cpu")
        while listening:

            time.sleep(0.1)

            while not requests_queue.empty():
                request = requests_queue.get()
                # print(request)
                self.BitVector[self.tensor_parallel_rank][self.rank] = compress(request[3][1])
                request_func[0] = request[1]
                request_args[0] = request[2]
                request_group[0] = request[3]

            while not p2p_queue.empty():
                request = p2p_queue.get()
                for i in range(len(request)):
                    self.BitVector[num_threads+1+self.tensor_parallel_rank][self.rank] += (1 << request[i].peer)
                p2p_request_args[0] = request
                p2p_finished_thread[0] = 1
    
            tmp_BitVector = self.BitVector.clone()
            if global_request_func is not None:
                tmp_BitVector[num_threads][self.rank] = compress(global_request_group[1])
            
            dist.all_reduce(tmp_BitVector, ReduceOp.SUM, self._CONTROLLER_GROUP)
            # print(tmp_BitVector)

            if tmp_BitVector[num_threads][self.rank] != 0:
                if is_ready(tmp_BitVector[num_threads], global_request_group[1]):
                    with global_condition:
                        # print(dist.get_rank(), global_request_args, global_request_group)
                        global_request_func(*global_request_args,global_request_group[0])
                        # print(global_request_func, 'finished')
                        global_request_func = None
                        global_condition.notify()
            
            if self.BitVector[self.tensor_parallel_rank][self.rank] != 0 and is_ready_backward(self.tensor_parallel_rank,tmp_BitVector,request_group[0][1]):
                # print('dealing with', tmp_BitVector, request_group[0][1])
                with request_conditions[0]:
                    request_func[0](*request_args[0], request_group[0][0])
                    # print('finished')
                    self.BitVector[self.tensor_parallel_rank][self.rank] = 0
                    finished_thread[0] = 0
                    request_conditions[0].notify()

            world_size = dist.get_world_size()
            if p2p_finished_thread[0]:
                visited = [0 for i in range(0, world_size)]
                graph = tmp_BitVector[num_threads+1+self.tensor_parallel_rank].cpu()
                if DFS(self.rank, world_size):
                    # print('dealing with', dist.get_rank(), tmp_BitVector)
                    self.BitVector[num_threads+1+self.tensor_parallel_rank][self.rank] = 0
                    p2p_finished_thread[0] = 0
                    # print('!!!!!!!!!!!!!!!!!!!!!!!!')
                    with p2p_conditions[0]:
                        # start_time = time.time()
                        p2p_reqs_list[0] = dist.batch_isend_irecv(p2p_request_args[0])
                        # for req in p2p_reqs_list[0]:
                        #     req.wait()
                        p2p_request_args[0] = None
                        p2p_conditions[0].notify()
                        # end_time = time.time()
                        # print('dealing with', dist.get_rank(), end_time-start_time)
        
    def get_status(self, index):
        return finished_thread[0]
    

class Communication_Controller (threading.Thread):
    def __init__(self, _CONTROLLER_GROUP):
        threading.Thread.__init__(self)
        self.result = None
        self._CONTROLLER_GROUP = _CONTROLLER_GROUP
        # print(dist.get_rank(), dist.get_backend(self._CONTROLLER_GROUP))
        self.BitVector = torch.zeros([2*num_threads+1, dist.get_world_size()], dtype = int)
        self.rank = dist.get_rank()

    def run(self):
        global listening
        global request_args
        global request_func
        global request_group
        global global_request_func
        global global_request_args
        global global_request_group
        global visited
        global graph
        # torch.device("cpu")
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

            # time.sleep(0.01)

            while not requests_queue.empty():
                request = requests_queue.get()
                # print(request[0], self.rank, request[3][1])
                self.BitVector[request[0]][self.rank] = compress(request[3][1])
                request_func[request[0]] = request[1]
                request_args[request[0]] = request[2]
                request_group[request[0]] = request[3]

            while not p2p_queue.empty():
                request = p2p_queue.get()
                for i in range(len(request[1])):
                    # if is_forward_rank(request[1][i].peer):
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
            # print('controller running, state:')
            # print(tmp_BitVector)
            # start_time = time.time()
            # print()
            if tmp_BitVector[num_threads][self.rank] != 0:
                if is_ready(tmp_BitVector[num_threads],global_request_group[1]):
                    # print(global_request_func, global_request_args)
                    # print('dealing with global', global_request_func, tmp_BitVector, global_request_group)
                    with global_condition:
                        # print(dist.get_rank(), global_request_args, global_request_group)
                        global_request_func(*global_request_args,global_request_group[0])
                        # print(dist.get_rank(), global_request_args)
                        # print(global_request_func, 'finished')
                        global_request_func = None
                        global_condition.notify()
                # print('after:',tmp_BitVector)
            for i in range(num_threads):
                if self.BitVector[i][self.rank] != 0 and is_ready(tmp_BitVector[i],request_group[i][1]):
                    # print('dealing with', i, tmp_BitVector, request_group[i][1])
                    with request_conditions[i]:
                        request_func[i](*request_args[i], request_group[i][0])
                        # print('finished', i)
                        self.BitVector[i][self.rank] = 0
                        finished_thread[i] = 0
                        request_conditions[i].notify()
            
            world_size = dist.get_world_size()

            for i in range(num_threads+1,num_threads*2+1):
                index = i - num_threads - 1
                if p2p_finished_thread[index]:
                    visited = [0 for i in range(0, world_size)]
                    graph = tmp_BitVector[i].cpu()
                    # start_time = time.time()
                    if DFS(self.rank, world_size):
                        # end_time = time.time()
                        # print('controller time:',i,end_time-start_time,tmp_BitVector)
                        # for x in p2p_request_args[index]:
                        #     print(dist.get_rank(),':',x.tensor.shape)
                        self.BitVector[i][self.rank] = 0
                        p2p_finished_thread[index] = 0
                        # print('!!!!!!!!!!!!!!!!!!!!!!!!')
                        # start_time = time.time()
                        torch.cuda.synchronize()
                        with p2p_conditions[index]:
                            # print('dealing with', dist.get_rank(), tmp_BitVector, p2p_request_args[index])
                            # print(p2p_request_args[index][0].tensor.device, p2p_request_args[index][0].tensor.is_contiguous())
                            # P2POp = dist.P2POp(p2p_request_args[index][0].op, p2p_request_args[index][0].tensor, p2p_request_args[index][0].peer)

                            p2p_reqs_list[index] = dist.batch_isend_irecv(p2p_request_args[index])

                            # p2p_reqs_list[index] = dist.batch_isend_irecv([P2POp])

                            # print(dist.get_rank(), 'finished')
                            p2p_request_args[index] = None
                            p2p_conditions[index].notify()
            # end_time = time.time()
            # print('controller time:',end_time-start_time,tmp_BitVector)
                        # end_time = time.time()
                        # print('batch_isend_irecv',dist.get_rank(),end_time-start_time,tmp_BitVector)
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
    # print(dist.get_rank(),rank_info)

def init(num_t, _CONTROLLER_GROUP, group_ranks, _GLOO_GROUP, tensor_rank):
    global _GLOBAL_GROUP
    global _CONTROLLER_GROUP_RANKS
    global _GLOBAL_GROUP_GLOO
    global normal_communication
    _GLOBAL_GROUP = _CONTROLLER_GROUP
    _CONTROLLER_GROUP_RANKS = group_ranks
    _GLOBAL_GROUP_GLOO = _GLOO_GROUP
    normal_communication = False

    global thread_barrier
    thread_barrier = threading.Barrier(num_t, timeout=None)

    global tensor_parallel_rank
    tensor_parallel_rank = tensor_rank

    global num_threads
    num_threads = num_t
    # global num_forward_ranks
    # num_forward_ranks = total_num_forward_ranks
    # global thread_barrier
    # thread_barrier = threading.Barrier(num_threads, timeout=None)
    # global use_thread_communication
    # use_thread_communication = True
    # normal_communication = False
    # global listening
    # listening = True
    # global request_args
    # global request_func
    # global request_group
    # global request_conditions
    # global p2p_reqs_list
    # global p2p_request_args
    # global p2p_conditions
    # request_args = [None for i in range(0, num_threads)]
    # request_func = [None for i in range(0, num_threads)]
    # request_group = [None for i in range(0, num_threads)]
    # request_conditions = [threading.Condition() for i in range(0, num_threads)]
    # p2p_reqs_list = [None for i in range(0, num_threads)]
    # p2p_request_args = [None for i in range(0, num_threads)]
    # p2p_conditions = [threading.Condition() for i in range(0, num_threads)]
    # global global_request_func
    # global global_request_args
    # global global_condition
    # global_request_args = None
    # global_request_func = None
    # global_condition = threading.Condition()
    # global finished_thread
    # global p2p_finished_thread
    # finished_thread = [0 for i in range(0, num_threads)]
    # p2p_finished_thread = [0 for i in range(0, num_threads)]
    # global _FORWARD_CONTROLLER_GROUP
    # global _CONTROLLER_GROUP_RANKS
    # _FORWARD_CONTROLLER_GROUP = _CONTROLLER_GROUP
    # _CONTROLLER_GROUP_RANKS = group_ranks
    # print('initiallize:',group_ranks)

def init_backward(total_num_threads, _CONTROLLER_GROUP, group_ranks, tensor_parallel_rank):
    global num_threads
    num_threads = total_num_threads
    global global_condition
    global_condition = threading.Condition()
    global normal_communication
    normal_communication = False
    global request_args
    global request_func
    global request_group
    global request_conditions
    request_args = [None for i in range(0, num_threads)]
    request_func = [None for i in range(0, num_threads)]
    request_group = [None for i in range(0, num_threads)]
    request_conditions = [threading.Condition() for i in range(0, num_threads)]
    global p2p_reqs_list
    global p2p_request_args
    global p2p_conditions
    p2p_reqs_list = [None for i in range(0, num_threads)]
    p2p_request_args = [None for i in range(0, num_threads)]
    p2p_conditions = [threading.Condition() for i in range(0, num_threads)]
    global finished_thread
    global p2p_finished_thread
    finished_thread = [0 for i in range(0, num_threads)]
    p2p_finished_thread = [0 for i in range(0, num_threads)]
    global controller
    global _CONTROLLER_GROUP_RANKS
    controller = Backward_Controller(_CONTROLLER_GROUP, tensor_parallel_rank)
    _CONTROLLER_GROUP_RANKS = group_ranks

def start_backward_controller():
    global listening
    listening = True
    controller.start()

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
    global use_thread_communication
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
    if normal_communication:
        return dist.barrier(group)
    global global_request_func
    global global_request_args
    global global_request_group
    # print(dist.get_rank(), 'attend')
    if not use_thread_communication:
        if group is None:
            group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
        perform_normal_func(dist.barrier,(group[0]), group[1])
    else:
        if get_thread_index() == 0 or get_thread_index() == -1:
            if group is None:
                group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
            perform_normal_func(dist.barrier,(group[0]), group[1])
        # global controller
        # print(dist.get_rank(),controller.is_alive())
        if get_thread_index() != -1:
            thread_barrier.wait()

def tensor_parallel_barrier():
    thread_barrier.wait()

def get_world_size(group = None,):
    if normal_communication:
        return dist.get_world_size(group)
    if isinstance(group, list):
        return dist.get_world_size(group[0])
    elif isinstance(group, int):
        return num_threads
    elif group is None:
        if _GLOBAL_RANK_INFO is None:
            return dist.get_world_size() - 1
        else:
            return len(_GLOBAL_RANK_INFO)
    else:
        return dist.get_world_size(group)

def get_rank(group = None,):
    # start_time = time.time()
    if normal_communication:
        return dist.get_rank(group)
    if isinstance(group, list):
        res = dist.get_rank(group[0])
    elif isinstance(group, int):
        res = get_thread_index()
    elif group is None:
        res = get_virtual_rank(dist.get_rank() - 1)
    else:
        res = dist.get_rank(group)
    # end_time = time.time()
    # print(dist.get_rank(),"get_rank ~ duration",end_time-start_time)
    return res

def init_process_group(backend, world_size, rank, timeout,):
    dist.init_process_group(backend = backend, world_size = world_size, rank = rank, timeout = timeout)

def set_thread_index(rank):
    global thread_mappings
    # print('%', threading.get_ident(), rank)
    thread_mappings[threading.get_ident()] = rank

def Global_Adjust(group_ranks):
    res = []
    las = -10
    for rank in group_ranks:
        if not (_GLOBAL_RANK_INFO[rank] > 0 and rank == las + 1 and _GLOBAL_RANK_INFO[rank] == _GLOBAL_RANK_INFO[las] + 1):
            res.append(rank)
        las = rank
    return res

def perform_normal_func(distfunc, distfuncargs, group_ranks):
    world_size = get_world_size()
    send_buffer = torch.zeros(world_size * 2 + 1, dtype = int)
    group_ranks = Global_Adjust(group_ranks)
    # print(group_ranks, world_size)
    for i in group_ranks:
        send_buffer[i] = 1
    rank = get_rank()
    send_buffer[world_size * 2] = rank
    # print(rank, 'attend')
    # print(distfunc, distfuncargs, group_ranks, dist.get_rank())
    # print('src:', dist.get_rank(), 'tag:', _GLOBAL_RANK_INFO[rank] + world_size, _GLOBAL_GROUP_GLOO)
    dist.send(tensor = send_buffer, dst = 0, tag = _GLOBAL_RANK_INFO[rank] + world_size, group = _GLOBAL_GROUP_GLOO)
    # print(send_buffer)
    # print('?')
    dist.recv(tensor = dummy, src = 0, tag = _GLOBAL_RANK_INFO[rank]+ world_size, group = _GLOBAL_GROUP_GLOO)
    if isinstance(distfuncargs, tuple):
        distfunc(*distfuncargs)
    else:
        distfunc(distfuncargs)
    # print('done', dist.get_rank())
    dist.send(tensor = dummy, dst = 0, tag = _GLOBAL_RANK_INFO[rank] + world_size, group = _GLOBAL_GROUP_GLOO)

def perform_p2p_func(distfunc, p2p_op_list):
    world_size = get_world_size()
    send_buffer = torch.zeros(world_size * 2 + 1, dtype = int)
    rank = get_rank()
    for p2p_op in p2p_op_list:
        send_buffer[get_virtual_rank(p2p_op.peer - 1) + world_size] = 1
    send_buffer[world_size * 2] = rank
    # print(rank, [x.peer for x in p2p_op_list], send_buffer)
    import time
    start_time = time.time()
    dist.send(tensor = send_buffer, dst = 0, tag = _GLOBAL_RANK_INFO[rank] + world_size, group = _GLOBAL_GROUP_GLOO)
    dist.recv(tensor = dummy, src = 0, tag = _GLOBAL_RANK_INFO[rank] + world_size, group = _GLOBAL_GROUP_GLOO)
    reqs = distfunc(p2p_op_list)
    dist.send(tensor = dummy, dst = 0, tag = _GLOBAL_RANK_INFO[rank] + world_size, group = _GLOBAL_GROUP_GLOO)
    end_time = time.time()
    # print('p2p time:', dist.get_rank(), end_time-start_time)
    return reqs

def batch_isend_irecv(p2p_op_list, bypass_controller = False):
    if normal_communication:
        return dist.batch_isend_irecv(p2p_op_list)
    # lock.acquire()
    # global listening
    # tmp = listening
    # if tmp:
    #     stop_controller()
    # print('stopped',dist.get_rank())
    if not use_thread_communication:
        # start_time = time.time()
        # print('##############',dist.get_rank())
        # if controller is not None:
        #     controller.set_p2p_state(p2p_op_list, 1)
        # print(dist.get_rank(), p2p_op_list)
        # start_time = time.time()
        # print('batch_isend_irecv', dist.get_rank(), p2p_op_list[0].peer, bypass_controller)
        if bypass_controller:
            reqs = dist.batch_isend_irecv(p2p_op_list)
        else:
            reqs = perform_p2p_func(dist.batch_isend_irecv, p2p_op_list)
        # end_time = time.time()
        # print('batch_isend_irecv', dist.get_rank(), end_time-start_time)
        # if controller is not None:
        #     controller.set_p2p_state(p2p_op_list, -1)
        # print('fi?? on', dist.get_rank())
        # end_time = time.time()
        # print('batch_isend_irecv', dist.get_rank(), end_time-start_time)
        # print('fibatch_isend_irecv')
        return reqs
    
    # print('batch_isend_irecv start',dist.get_rank())
    # start_time = time.time()
    reqs = perform_p2p_func(dist.batch_isend_irecv, p2p_op_list)
    # end_time = time.time()
    # print('batch_isend_irecv', dist.get_rank(), end_time-start_time)
    # reqs = dist.batch_isend_irecv(p2p_op_list)
    # if tmp:
    #     start_controller()
    # print('batch_isend_irecv finished',dist.get_rank())
    # lock.release()
    return reqs

def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    if normal_communication:
        return dist.all_reduce(tensor, op, group, async_op)
    if donothing:
        if async_op:
            return Handle()
        return
    global use_thread_communication
    global result
    global global_request_func
    global global_request_args
    global global_request_group
    # print('all_reduce start')
    # print('all_reduce start', dist.get_rank(), group, use_thread_communication)
    if not use_thread_communication:
        # print('all_reduce start')
        # start_time = time.time()
        if group is None:
            group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
        # print('all_reduce start', dist.get_rank(), backward_rank_only(group[1]))
        if backward_rank_only(group[1]):
            # print('&&')
            # if dist.get_rank() == 5:
            #     end_time = time.time()
            #     print("all_reduce ~ duration",end_time-start_time)
            # print('all_reduce start', dist.get_rank())
            req = dist.all_reduce(tensor, op, group[0], async_op)
            # print('all_reduce finish', dist.get_rank())
            return req
        else:
            perform_normal_func(dist.all_reduce,(tensor, op, group[0]), group[1])
        # if dist.get_rank() == 5:
        #     if group is not None:
        #         print('??????', group[1])
        # handle = dist.all_reduce(tensor, op, group, async_op)
        
        # print('all_reduce finished')
    elif isinstance(group, int):
        thread_barrier.wait()
        # start_time = time.time()
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
        # end_time = time.time()
        # print(dist.get_rank(),"all_reduce ! duration",end_time-start_time)
    elif group is None or len(group) == 3:
        # start_time = time.time()
        if group is None:
            group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
        if get_thread_index() == -1:
            with global_condition:
                global_request_group = group
                global_request_args = (tensor, op)
                global_request_func = dist.all_reduce
                while global_request_func is not None:
                    global_condition.wait()
        else:
            # print('all_reduce', get_thread_index(), get_rank(), _CONTROLLER_GROUP_RANKS)
            # print('#############')
            lock.acquire()
            # print(tensor)
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
            # print(get_thread_index(), num_threads)
            if get_thread_index() == 0:
                perform_normal_func(dist.all_reduce,(tensor, op, group[0]), group[1])
            # print('?', get_thread_index(), num_threads)
            thread_barrier.wait()
            # print(get_thread_index(), tensor, result)
            # print(result,get_thread_index())
            # print(result)
            tensor.copy_(result)
            thread_barrier.wait()
            result = None
            thread_barrier.wait()
        # print('all_reduce', get_rank(),'finished')
        # end_time = time.time()
        # print(dist.get_rank(),"all_reduce @ duration",end_time-start_time)
    else:
        # start_time = time.time()
        perform_normal_func(dist.all_reduce,(tensor, op, group[0]), group[1])
        # print('ed', index)
        # end_time = time.time()
        # print(dist.get_rank(),"all_reduce # duration",end_time-start_time)
        # print(group)
    # print('all_reduce finished')
    if async_op:
        return Handle()

def get_global_shape(tensor):
    new_tensor = tensor.repeat(num_threads)
    return new_tensor

def Recover(tensor_list, result, length):
    s = 0
    cur_id = 0
    for i in range(0, dist.get_world_size() - 1):
        if _GLOBAL_RANK_INFO[s] == 0:
            tensor_list[cur_id:cur_id+length*num_threads]=result[s*length:s*length+length*num_threads]
            cur_id += length*num_threads
            s += num_threads
        else:
            tensor_list[cur_id:cur_id+length]=result[s*length:s*length+length]
            cur_id += length
            s += 1

def _all_gather_base(tensor_list, tensor, group=None, async_op=False):
    if normal_communication:
        return dist._all_gather_base(tensor_list, tensor, group, async_op)
    global use_thread_communication
    global new_tensor
    global result
    global global_request_func
    global global_request_args
    global global_request_group
    # print('all_gather start')
    if not use_thread_communication:
        # print('all_gather start')
        # if group is None:
        #     result = get_global_shape(tensor)
        #     # print("all_gather", dist.get_rank(), result)for
        #     handle = dist.all_gather(result, tensor, group, async_op)
        #     # print("all_gather", dist.get_rank(), "finished")
        #     tensor_list.copy_(result.view(-1))
        # else:
        #     # start_time = time.time()
        #     handle = dist._all_gather_base(tensor_list, tensor, group, async_op)
        #     # end_time = time.time()
        #     # print(dist.get_rank(),"_all_gather_base ~ duration",end_time-start_time)
        # # print('all_gather finished')
        if group is None:
            group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
            new_tensor = get_global_shape(tensor)
            result = new_tensor.repeat(dist.get_world_size() - 1)
            # print(dist.get_rank(), result)
            # with global_condition:
            #     global_request_group = group
            #     global_request_args = (result, new_tensor)
            #     global_request_func = dist._all_gather_base
            #     while global_request_func is not None:
            #         global_condition.wait()
            perform_normal_func(dist._all_gather_base, (result, new_tensor, group[0]), group[1])
            Recover(tensor_list, result, tensor.shape[0])
        elif len(group) == 3:
            return dist._all_gather_base(tensor_list, tensor, group, async_op)
        else:
            perform_normal_func(dist._all_gather_base, (tensor_list, tensor, group[0]), group[1])
        # if dist.get_rank() == 5:
        #     print('??????')
    elif isinstance(group, int):
        # start_time = time.time()
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
        # end_time = time.time()
        # print(dist.get_rank(),"_all_gather_base @ duration",end_time-start_time)
    elif group is None:
        group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
        lock.acquire()
        if new_tensor is None:
            new_tensor = get_global_shape(tensor).view(-1)
            result = new_tensor.repeat(dist.get_world_size() - 1)
            # print(dist.get_rank(), result)
        thread_id = get_thread_index()
        new_tensor[thread_id*tensor.shape[0]:(thread_id+1)*tensor.shape[0]] = tensor.clone()
        lock.release()
        thread_barrier.wait()
        if get_thread_index() == 0:
            perform_normal_func(dist._all_gather_base, (result, new_tensor, group[0]), group[1])
        thread_barrier.wait()
        Recover(tensor_list, result, tensor.shape[0])
        thread_barrier.wait()
        result = None
        new_tensor = None
        thread_barrier.wait()
    else:
        perform_normal_func(dist._all_gather_base, (tensor_list, tensor, group[0]), group[1])
    # print('all_gather finished')
    if async_op:
        return Handle()

def Recover_object_list(result, object_list):
    object_list = []
    for obj_list in result:
        for obj in obj_list:
            object_list.append(obj)
    return object_list

def all_gather_object(object_list, obj, group=None):
    if normal_communication:
        return dist.all_gather_object(object_list, obj, group=None)
    global use_thread_communication
    global result
    global new_tensor
    if not use_thread_communication:
        if group is None:
            group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
            result = [None for i in range(0, dist.get_world_size()-1)]
            perform_normal_func(dist.all_gather_object, (result, [obj], group[0]), group[1])
            Recover_object_list(result, object_list)
        elif len(group) == 3:
            return dist.all_gather_object(object_list, obj, group)
        else:
            perform_normal_func(dist.all_gather_object, (object_list, [obj], group[0]), group[1])
    elif isinstance(group, int):
        lock.acquire()
        if result is None:
            result = []
        result.append((get_thread_index(),obj))
        lock.release()
        thread_barrier.wait()
        for x in result:
            object_list[x[0]]=x[1].clone()
        thread_barrier.wait()
        result = None
        thread_barrier.wait()
    elif group is None:
        group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
        lock.acquire()
        thread_id = get_thread_index()
        if new_tensor is None:
            new_tensor = [None for i in range(0, num_threads)]
            new_tensor[thread_id] = obj
            result = [None for i in range(0, dist.get_world_size()-1)]
            # print(dist.get_rank(), result)
        new_tensor[thread_id] = obj
        lock.release()
        thread_barrier.wait()
        if get_thread_index() == 0:
            perform_normal_func(dist.all_gather_object, (result, new_tensor, group[0]), group[1])
        thread_barrier.wait()
        Recover_object_list(result, object_list)
        thread_barrier.wait()
        result = None
        new_tensor = None
        thread_barrier.wait()
        # raise NotImplementedError("TODO: all_gather_object global comm")
        # lock.acquire()
        #     # print(dist.get_rank(), result)
        # thread_id = get_thread_index()
        # new_tensor[thread_id*tensor.shape[0]:(thread_id+1)*tensor.shape[0]] = tensor.clone()
        # lock.release()
        # thread_barrier.wait()
        # if get_thread_index() == 0:
        #     with global_condition:
        #         global_request_group = [None, _CONTROLLER_GROUP_RANKS]
        #         global_request_args = (result, new_tensor)
        #         global_request_func = dist._all_gather_base
        #         while global_request_func is not None:
        #             global_condition.wait()
        # thread_barrier.wait()
        # Recover(tensor_list, result, tensor.shape[0])
        # thread_barrier.wait()
        # result = None
        # new_tensor = None
        # thread_barrier.wait()
    else:
        perform_normal_func(dist.all_gather_object, (object_list, obj, group[0]), group[1])

def gather_object(obj, object_list, dst=0, group=None):
    if normal_communication:
        return gather_object(obj, object_list, dst, group)
    global use_thread_communication
    global result
    global new_tensor
    if not use_thread_communication:
        if group is None:
            group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
            result = [None for i in range(0, dist.get_world_size()-1)]
            perform_normal_func(dist.all_gather_object, (result, [obj], group[0]), group[1])
            Recover_object_list(result, object_list)
        elif len(group) == 3:
            return dist.all_gather_object(object_list, obj, group)
        else:
            perform_normal_func(dist.all_gather_object, (object_list, [obj], group[0]), group[1])
    elif isinstance(group, int):
        lock.acquire()
        if result is None:
            result = []
        result.append((get_thread_index(),obj))
        lock.release()
        thread_barrier.wait()
        for x in result:
            object_list[x[0]]=x[1].clone()
        thread_barrier.wait()
        result = None
        thread_barrier.wait()
    elif group is None:
        group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
        lock.acquire()
        thread_id = get_thread_index()
        if new_tensor is None:
            new_tensor = [None for i in range(0, num_threads)]
            new_tensor[thread_id] = obj
            result = [None for i in range(0, dist.get_world_size()-1)]
            # print(dist.get_rank(), result)
        new_tensor[thread_id] = obj
        lock.release()
        thread_barrier.wait()
        if get_thread_index() == 0:
            perform_normal_func(dist.all_gather_object, (result, new_tensor, group[0]), group[1])
        thread_barrier.wait()
        Recover_object_list(result, object_list)
        thread_barrier.wait()
        result = None
        new_tensor = None
        thread_barrier.wait()
        # raise NotImplementedError("TODO: all_gather_object global comm")
        # lock.acquire()
        #     # print(dist.get_rank(), result)
        # thread_id = get_thread_index()
        # new_tensor[thread_id*tensor.shape[0]:(thread_id+1)*tensor.shape[0]] = tensor.clone()
        # lock.release()
        # thread_barrier.wait()
        # if get_thread_index() == 0:
        #     with global_condition:
        #         global_request_group = [None, _CONTROLLER_GROUP_RANKS]
        #         global_request_args = (result, new_tensor)
        #         global_request_func = dist._all_gather_base
        #         while global_request_func is not None:
        #             global_condition.wait()
        # thread_barrier.wait()
        # Recover(tensor_list, result, tensor.shape[0])
        # thread_barrier.wait()
        # result = None
        # new_tensor = None
        # thread_barrier.wait()
    else:
        perform_normal_func(dist.all_gather_object, (object_list, obj, group[0]), group[1])

def _reduce_scatter_base(tensor, tensor_list, op=ReduceOp.SUM, group=None, async_op=False):
    if normal_communication:
        return dist._reduce_scatter_base(tensor, tensor_list, op, group, async_op)
    global use_thread_communication
    if not use_thread_communication:
        # start_time = time.time()
        # return dist._reduce_scatter_base(tensor, tensor_list, op, group, async_op)
        # end_time = time.time()
        # print(dist.get_rank(),"_reduce_scatter_base ~ duration",end_time-start_time)
        if group is None:
            group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
        if len(group) == 3:
            return dist._reduce_scatter_base(tensor, tensor_list, op, group, async_op)
        else:
            perform_normal_func(dist._reduce_scatter_base, (tensor, tensor_list, op, group[0]), group[1])
        # if dist.get_rank() == 5:
        #     print('??????')
    elif isinstance(group, int):
        # start_time = time.time()
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
        # end_time = time.time()
        # print(dist.get_rank(),"_reduce_scatter_base @ duration",end_time-start_time)
    else:
        # start_time = time.time()
        # print('st')
        perform_normal_func(dist._reduce_scatter_base, (tensor, tensor_list, op, group[0]), group[1])
        # print('ed')
        # end_time = time.time()
        # print(dist.get_rank(),"_reduce_scatter_base # duration",end_time-start_time)
    if async_op:
        return Handle()

def broadcast(tensor, src, group=None):
    global use_thread_communication
    global global_request_func
    global global_request_args
    global global_request_group
    global result
    if normal_communication:
        return dist.broadcast(tensor, src, group)
    # print('$', src, group)
    if not use_thread_communication:
        # start_time = time.time()
        src = get_real_rank(src)
        # print('broadcast start', dist.get_rank(), src, group)
        # dist.broadcast(tensor, src, group)
        # print('broadcast finished')
        # end_time = time.time()
        # print(dist.get_rank(),"broadcast ! duration",end_time-start_time)
        if group is None:
            group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
        if len(group) == 3:
            # print('broadcast start', dist.get_rank(), src)
            req = dist.broadcast(tensor, src, group[0])
            # print('broadcast end', dist.get_rank(), src)
            # if dist.get_rank() == 5:
            #     end_time = time.time()
            #     print('broadcast', end_time-start_time)
            return req
        else:
            perform_normal_func(dist.broadcast, (tensor, src, group[0]), group[1])
        # print('broadcast done')
    elif group is None:
        group = [_GLOBAL_GROUP, _CONTROLLER_GROUP_RANKS]
        # src = get_real_rank(src)
        target = max(_GLOBAL_RANK_INFO[src], 0)
        if get_thread_index() == target:
            perform_normal_func(dist.broadcast, (tensor, get_real_rank(src), group[0]), group[1])
            result = tensor
        thread_barrier.wait()
            # print(result,get_thread_index())
            # print(result)
        tensor.copy_(result)
        thread_barrier.wait()
        result = None
        thread_barrier.wait()
        # global_request_group = (None, _CONTROLLER_GROUP_RANKS)
        # global_request_args = (tensor, src, group)
        # global_request_func = dist.broadcast
        # with global_condition:
        #     while global_request_func is not None:
        #         global_condition.wait()
    elif isinstance(group, int):
        # print('@', src, _GLOBAL_RANK_INFO)
        # start_time = time.time()
        if get_thread_index() == _GLOBAL_RANK_INFO[src]:
            result = tensor
        thread_barrier.wait()
        tensor.copy_(result)
        thread_barrier.wait()
        result = None
        thread_barrier.wait()
        # end_time = time.time()
        # print(dist.get_rank(),"broadcast # duration",end_time-start_time)
    else:
        # start_time = time.time()
        src = get_real_rank(src)
        get_real_rank(src)
        # print(group)
        perform_normal_func(dist.broadcast, (tensor, src, group[0]), group[1])
        # end_time = time.time()
        # print(dist.get_rank(),"broadcast @ duration",end_time-start_time)