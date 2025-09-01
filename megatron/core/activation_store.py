import torch
import megatron.virtual_tensor_parallel_communication as dist
from megatron.core import parallel_state

class ActivationSet:

    def __init__(self, ):
        self.activations = []
        self.id = []
        self.num = 0

    def store_activation(self, activation):
        self.num += 1
        if isinstance(activation, tuple):
            for element in activation:
                self.activations.append(element)
                if element is None:
                    self.id.append(-self.num)
                else:
                    self.id.append(self.num)
        else:
            self.activations.append(activation)
            if activation is None:
                self.id.append(-self.num)
            else:
                self.id.append(self.num)

    def load_activation(self, ):
        activation = self.activations.pop(0)
        return activation
    
    def send_coresponding_activations(self, config):
        from megatron.core.pipeline_parallel import p2p_communication
        num = torch.tensor([len(self.activations)], device=torch.cuda.current_device(), dtype=int)
        p2p_communication.send_corresponding_forward(num, config, True)
        p2p_communication.send_corresponding_forward(torch.tensor(self.id, device=torch.cuda.current_device(), dtype=int), config, True)
        for activation in self.activations:
            if activation is not None:
                shape = torch.tensor(activation.shape, device=torch.cuda.current_device(), dtype=int)
                p2p_communication.send_corresponding_forward(torch.tensor([len(shape)], device=torch.cuda.current_device(), dtype=int), config, True)
                p2p_communication.send_corresponding_forward(shape, config, True)
        for activation in self.activations:
            if activation is not None:
                p2p_communication.send_corresponding_forward(activation, config, True)

    def recv_coresponding_activations(self, config):
        from megatron.core.pipeline_parallel import p2p_communication
        num = torch.empty(1, device=torch.cuda.current_device(), dtype=int)
        num = p2p_communication.recv_corresponding_forward(num.shape, config, dtype=int, bypass_controller=True)
        id = torch.empty(num.item(), device=torch.cuda.current_device(), dtype=int)
        id = p2p_communication.recv_corresponding_forward(id.shape, config, dtype=int, bypass_controller=True)
        shapes = []
        for i in range(num.item()):
            if id[i] > 0:
                ndim = torch.empty(1, device=torch.cuda.current_device(), dtype=int)
                ndim = p2p_communication.recv_corresponding_forward(ndim.shape, config, dtype=int, bypass_controller=True)
                shape = torch.empty(ndim.item(), device=torch.cuda.current_device(), dtype=int)
                shape = p2p_communication.recv_corresponding_forward(shape.shape, config, dtype=int, bypass_controller=True)
                shapes.append(tuple(shape.tolist()))
            else:
                shapes.append(0)
        elements = []
        for i in range(num.item()):
            if id[i] > 0:
                element = torch.empty(shapes[i], dtype=config.params_dtype)
                element = p2p_communication.recv_corresponding_forward(element.shape, config, bypass_controller=True)
            else:
                element = None
            elements.append(element)
        activation = []
        for i in range(num.item()):
            activation.append(elements[i])
            if i == num.item() - 1 or abs(id[i]) != abs(id[i+1]):
                if len(activation) == 1:
                    self.activations.append(activation[0])
                else:
                    self.activations.append(tuple(activation))
                activation = []

    def reset(self, ):
        self.activations = []

TensorStore = None
TensorStoreSets = None

def init_sets():
    global TensorStore
    global TensorStoreSets
    if parallel_state.is_forward_stage():
        TensorStore = [ActivationSet() for i in range(dist.num_threads)]
    else:
        TensorStoreSets = []
    # print(dist.get_rank(), TensorStore)

def store_activation(activation):
    global TensorStore
    # print(dist.get_rank(), TensorStore)
    TensorStore[dist.get_thread_index()].store_activation(activation)

def load_activation():
    global TensorStoreSets
    return TensorStoreSets[0].load_activation()

def send_activations(config):
    global TensorStore
    # print('sending')
    TensorStore[dist.get_thread_index()].send_coresponding_activations(config)
    TensorStore[dist.get_thread_index()].reset()

def recv_activations(config):
    global TensorStoreSets
    TensorStoreSets.append(ActivationSet())
    # print('recving')
    TensorStoreSets[-1].recv_coresponding_activations(config)

def next_set():
    global TensorStoreSets
    TensorStoreSets.pop(0)

