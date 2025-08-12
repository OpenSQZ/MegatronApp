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
import torch.distributed as dist
import time
import sys
import threading

def is_ready(BitVector, group):
    state = compress(group)
    for i in group:
        if BitVector[i] != state:
            return False
    return True

def normal_comm_check(normal_comm_t, rank, virtual_world_size):
    res = []
    for i in range(0, virtual_world_size):
        if normal_comm_t[rank][i]:
            res.append(i)
            for j in range(0, virtual_world_size):
                if normal_comm_t[rank][j] != normal_comm_t[i][j]:
                    return []
    return res

visited = None
graph = None
def DFS(rank, world_size):
    if visited[rank] == 1:
        return True
    visited[rank] = 1
    for i in range(0, world_size):
        if graph[rank][i]:
            if (not graph[i][rank]) or (not DFS(i, world_size)):
                return False
    return True

def p2p_comm_check(p2p_comm_t, rank, virtual_world_size):
    global graph
    global visited
    graph = p2p_comm_t
    visited = [0 for i in range(virtual_world_size)]
    # print(graph)
    if p2p_comm_t[rank].eq(0).all() or (not DFS(rank, virtual_world_size)):
        return []
    res = [i for i, val in enumerate(visited) if val == 1]
    return res

def wait_and_callback(i, req):
    if not req.is_completed():
        req.wait()
    # print(i, 'getting')

def start_server(virtual_world_size, _GLOBAL_GROUP_GLOO, REAL_RANK, _GLOBAL_RANK_INFO):
    rank = 0
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create one group per client (pairs: server and client)
    world_size = dist.get_world_size()

    # client_groups = {
    #     i: dist.new_group(ranks=[rank, i]) for i in range(1, world_size)
    # }
    for i in range(len(_GLOBAL_RANK_INFO)):
        _GLOBAL_RANK_INFO[i] += virtual_world_size

    recv_buffers  = [torch.zeros(virtual_world_size * 2 + 1, dtype = int) for i in range(0, virtual_world_size)]
    total = torch.zeros(virtual_world_size, virtual_world_size * 2, dtype = int)

    # global graph
    # graph = [torch.zeros(virtual_world_size, dtype = int) for i in range(0, virtual_world_size)]

    threads = []
    for i in range(0, virtual_world_size):
    #     print('src:', REAL_RANK[i], 'tag:', _GLOBAL_RANK_INFO[i])
        req = dist.irecv(tensor=recv_buffers[i], src = REAL_RANK[i], tag = _GLOBAL_RANK_INFO[i], group = _GLOBAL_GROUP_GLOO)
        t = threading.Thread(target=wait_and_callback, args=(i, req))
        t.start()
        threads.append(t)
    
    dummy = torch.zeros(1)
    # selfgroup = dist.new_group([0],backend='gloo')

    while True:
        # Check which recv requests have completed
        # dist.all_reduce(dummy, op=dist.ReduceOp.SUM, group=selfgroup)
        for i in range(0, virtual_world_size):
            # req = recv_reqs[i]
            # req.wait()
            # print('getting', i)
            # print('running')
            if threads[i] is not None and (not threads[i].is_alive()):
                # Copy received tensor and update stored tensor
                # stored_tensors[i] = recv_buffers.clone()
                # print(f"[Server] Received updated tensor from client {i}: {stored_tensors[i].tolist()}")

                # Repost new irecv to keep listening for next upload from this client
                total[recv_buffers[i][virtual_world_size * 2]] = recv_buffers[i][0 : virtual_world_size * 2]

                # Compute current aggregate sum
                
                normal_comm_t = total[:, 0:virtual_world_size]
                p2p_comm_t = total[:, virtual_world_size:2*virtual_world_size]

                # print(p2p_comm_t)
                # print('getting', i)
                # print(recv_buffers[i])
                # print(p2p_comm_t)

                ranks = normal_comm_check(normal_comm_t, i, virtual_world_size) + p2p_comm_check(p2p_comm_t, i, virtual_world_size)
                # print(normal_comm_t, ranks)

                for rank in ranks:
                    dist.send(tensor = dummy, dst = REAL_RANK[rank], tag = _GLOBAL_RANK_INFO[rank], group = _GLOBAL_GROUP_GLOO)

                for rank in ranks:
                    total[rank] = 0
                
                for rank in ranks:
                    dist.recv(tensor = dummy, src = REAL_RANK[rank], tag = _GLOBAL_RANK_INFO[rank], group = _GLOBAL_GROUP_GLOO)

                threads[i] = None
                for rank in ranks:
                    req = dist.irecv(tensor=recv_buffers[rank], src = REAL_RANK[rank], tag = _GLOBAL_RANK_INFO[rank], group = _GLOBAL_GROUP_GLOO)
                    t = threading.Thread(target=wait_and_callback, args=(i, req))
                    t.start()
                    threads[rank] = t
                # print(f"[Server] Current aggregated sum: {total.tolist()}")
