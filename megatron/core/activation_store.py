import torch
import megatron.virtual_tensor_parallel_communication as dist
from megatron.training.global_vars import get_args
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import p2p_communication
import megatron.virtual_tensor_parallel_communication as dist

class ActivationSet:

    def __init__(self, ):
        self.activations = []

    def store_activation(self, activation):
        self.activations.append(activation)

    def load_activation(self, ):
        activation = self.activations.pop(0)
        return activation
    
    def send_coresponding_activations(self, config):
        num = torch.tensor([len(self.activations)], dtype=int)
        p2p_communication.send_corresponding_forward(num, config)
        for activation in self.activations:
            shape = torch.tensor(activation.shape, dtype=int)
            p2p_communication.send_corresponding_forward(torch.tensor([len(shape)], dtype=torch.int), config)
            p2p_communication.send_corresponding_forward(shape, config)
        for activation in self.activations:
            p2p_communication.send_corresponding_forward(activation, config)

    def recv_coresponding_activations(self, config):
        num = torch.empty(1, dtype=int)
        p2p_communication.recv_corresponding_forward(num, config)
        shapes = []
        for i in range(num):
            ndim = torch.empty(1, dtype=int)
            p2p_communication.recv_corresponding_forward(ndim, config)
            shape = torch.empty(ndim.item(), dtype=int)
            p2p_communication.recv_corresponding_forward(shape, config)
            shapes.append(tuple(shape.tolist()))
        for i in range(num):
            activation = torch.empty(shapes[i], dtype=config.params_dtype)
            p2p_communication.recv_corresponding_forward(activation, config)
            self.activations.append(activation)

    def reset():
        self.activations = []

TensorStore = None
TensorStoreSets = None

def init_sets():
    if dist.is_forward_rank():
        TensorStore = [ActivationSet() for i in range(dist.num_threads)]
    else:
        TensorStoreSets = []

def store_activation(activation):
    TensorStore[dist.get_thread_index()].store_activation(activation)

def load_activation():
    return TensorStoreSets[0].load_activation(activation)

def send_activations(config):
    TensorStore[dist.get_thread_index()].send_coresponding_activations(config)
    TensorStore[dist.get_thread_index()].reset()

def recv_activations(config):
    TensorStoreSets.append(ActivationSet())
    TensorStoreSets[-1].recv_coresponding_activations(config)

def next_set():
    TensorStoreSets.pop(0)

