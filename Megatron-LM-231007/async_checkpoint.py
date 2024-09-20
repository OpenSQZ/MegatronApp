from threading import Thread
from queue import Queue
import time
import pickle
import torch
import torch.distributed as dist

from megatron import print_rank_0

queue = Queue()
is_killed = False
saved_args = None

def save_checkpoint(state_dict, file):
    global saved_args
    if saved_args.only_serialization:
        text = pickle.dumps(state_dict, protocol=saved_args.pickle_protocol)
        print(f'Rank {dist.get_rank()} will save {len(text)} bytes to {file}')
    elif saved_args.only_memcopy:
        text = state_dict.copy()
        print(f'Rank {dist.get_rank()} will save {len(text)} bytes to {file}')
    else:
        print(f'Rank {dist.get_rank()} will save checkpoint to {file}')
        torch.save(state_dict, file)

def commit_checkpoint(state_dict, file):
    global saved_args
    if saved_args.async_saving:
        queue.put((state_dict, file))
    else:
        save_checkpoint(state_dict, file)

def async_save_worker():
    global is_killed

    while not is_killed or not queue.empty():
        if queue.empty():
            time.sleep(0.01)
            continue
        state_dict, file = queue.get_nowait()
        save_checkpoint(state_dict, file)
    
    print_rank_0('Asynchronized saving worker has been killed')
        

def kill_thread():
    global is_killed
    is_killed = True

def init_checkpoint_handler(args):
    global saved_args
    saved_args = args
    if saved_args.async_saving:
        print_rank_0('Asynchronized saving enabled.')
        worker = Thread(target=async_save_worker)
        worker.start()
    else:
        print_rank_0('Asynchronized saving did not enable.')