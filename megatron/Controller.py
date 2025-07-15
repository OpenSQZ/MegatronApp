import torch
import torch.distributed as dist
import time
import sys

def start_server(virtual_world_size):
    rank = 0
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create one group per client (pairs: server and client)
    world_size = dist.get_world_size()

    client_groups = {
        i: dist.new_group(ranks=[rank, i]) for i in range(1, world_size)
    }

    recv_buffers  = [torch.zeros(world_size * 2 + 1, dtype = int) for i in range(1, world_size)]
    total = torch.zeros(virtual_world_size, world_size * 2)

    recv_reqs = []
    for i in range(0, world_size - 1):
        recv_reqs.append(dist.irecv(tensor=recv_buffers[i], src=i + 1))

    while True:
        # Check which recv requests have completed
        for i in range(0, world_size - 1):
            req = recv_reqs[i]
            if req.is_completed():
                # Copy received tensor and update stored tensor
                # stored_tensors[i] = recv_buffers.clone()
                # print(f"[Server] Received updated tensor from client {i}: {stored_tensors[i].tolist()}")

                # Repost new irecv to keep listening for next upload from this client
                total[recv_buffers[i][world_size * 2]] = recv_buffers[i][0 : world_size * 2]

                recv_reqs[i] = dist.irecv(tensor=recv_buffers[i], src = i + 1)

                # Compute current aggregate sum
                
                normal_comm_t = total[:][0:world_size]
                p2p_comm_t = total[:][world_size+1:2*world_size]

                Check(normal_comm_t, p2p_comm_t)
                # print(f"[Server] Current aggregated sum: {total.tolist()}")



def client_loop(rank, tensor_size=1):
    world_size = int(sys.argv[1])
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create group with server (rank 0) and self
    client_group = dist.new_group(ranks=[0, rank])

    # Initial tensor value for client
    val = float(rank)

    print(f"[Client {rank}] Started")

    while True:
        tensor = torch.full((tensor_size,), val)
        req = dist.isend(tensor=tensor, dst=0)
        req.wait()
        print(f"[Client {rank}] Sent tensor: {tensor.tolist()}")

        # Wait for server release signal (block here)
        print(f"[Client {rank}] Waiting at barrier")
        dist.barrier(group=client_group)
        print(f"[Client {rank}] Passed barrier, proceeding")

        # Update tensor value for next round (example logic)
        val += 1.0

        # Simulate some work before next upload
        time.sleep(2)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <world_size> <rank>")
        sys.exit(1)

    world_size = int(sys.argv[1])
    rank = int(sys.argv[2])

    if rank == 0:
        server_loop(world_size)
    else:
        client_loop(rank)