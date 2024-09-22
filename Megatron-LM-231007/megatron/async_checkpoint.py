from threading import Thread
from queue import Queue
import time
import pickle
import torch
import inc.torch as dist


from megatron import print_rank_0

queue = Queue()
is_killed = False
saved_args = None

def real_save_checkpoint(state_dict, file):
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
        real_save_checkpoint(state_dict, file)

def async_save_worker():
    global is_killed, queue

    while not is_killed or not queue.empty():
        if queue.empty():
            time.sleep(0.01)
            continue
        state_dict, file = queue.get_nowait()
        real_save_checkpoint(state_dict, file)
    
    print_rank_0('Asynchronized saving worker has been killed')


def kill_thread():
    global is_killed
    is_killed = True

inject = False
def start_injection():
    global inject
    inject = True

def start_experiment_thread(model, optimizer, opt_param_scheduler):
    print('Experiment thread called.')
    def worker_thread():
        from megatron_patch.checkpointing import save_checkpoint_experiment
        global is_killed, queue, inject
        print('Experiment thread started')
        cnt = 100000
        while not is_killed:
            if queue.empty() and inject:
                print('Experiment Commit saving')
                save_checkpoint_experiment(cnt, model, optimizer, opt_param_scheduler)
                cnt += 1
            time.sleep(0.01)
        print('Experiment thread stoped')
    worker = Thread(target=worker_thread)
    worker.start()

def init_checkpoint_handler(args):
    global saved_args
    saved_args = args
    if saved_args.async_saving:
        print_rank_0('Asynchronized saving enabled.')
        worker = Thread(target=async_save_worker)
        worker.start()
    else:
        print_rank_0('Asynchronized saving did not enable.')