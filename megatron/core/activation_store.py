import torch
import megatron.virtual_tensor_parallel_communication as dist
from megatron.training.global_vars import get_args
from megatron.core import parallel_state

class ActivationSet:

    def __init__(self, ):
        self.activations = []

    def store_activation(activation):
        self.activations.append(activation)

    def load_activation():
        self.activation = activations.pop(0)
        return activation
    
    def send_coresponding_activations():
        dst_rank = parallel_state.get_forward
        torch.stack(activations)