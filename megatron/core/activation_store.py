import torch
import megatron.virtual_tensor_parallel_communication as dist
from megatron.training.global_vars import get_args
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import p2p_communication

class ActivationSet:

    def __init__(self, ):
        self.activations = []

    def store_activation(activation):
        self.activations.append(activation)

    def load_activation():
        self.activation = activations.pop(0)
        return activation
    
    def send_coresponding_activations(config):
        for activation in self.activations:
            p2p_communication.send_corresponding_forward(activation, config)

    def recv_coresponding_activations(config):
        self.activations = get_shape()
        p2p_communication.send_corresponding_forward(self.activations, config)