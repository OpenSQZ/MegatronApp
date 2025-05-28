# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import os
import sys
import time
import csv
import torch
import json
import asyncio

from enum import Enum
from typing import Callable, Dict, Any

from megatron import dist_signal_handler
from megatron.tokenizer import build_tokenizer
from .microbatches import build_num_microbatches_calculator
from .timers import Timers
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_GLOBAL_ARGS = None
_GLOBAL_RETRO_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_TRACERS = None
_GLOBAL_SIGNAL_HANDLER = None
_GLOBAL_FLAGS = None
_GLOBAL_TRACING_GROUP = None
_GLOBAL_REPORT = lambda name, args, tensor: print(name,args) 
_GLOBAL_COMPRESSER = None
_GLOBAL_DISTURBANCE = None

mlp2_record = None

def set_filter():
    global mlp2_record
    mlp2_record = [None for i in range(get_args().num_layers + 1)]

def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def get_retro_args():
    """Return retro arguments."""
    return _GLOBAL_RETRO_ARGS


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples,
                                               consistency_check)


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_wandb_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_WANDB_WRITER


def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS

def get_tracers():
    """Return tracers."""
    _ensure_var_is_initialized(_GLOBAL_TRACERS, 'tracers')
    return _GLOBAL_TRACERS

def get_flags():
    """Return flags."""
    _ensure_var_is_initialized(_GLOBAL_FLAGS, 'flags')
    return _GLOBAL_FLAGS

def get_disturbance():
    _ensure_var_is_initialized(_GLOBAL_DISTURBANCE, 'change')
    return _GLOBAL_DISTURBANCE

def get_compresser():
    global _GLOBAL_COMPRESSER
    _ensure_var_is_initialized(_GLOBAL_COMPRESSER, "Compresser");
    return _GLOBAL_COMPRESSER

def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    return _GLOBAL_SIGNAL_HANDLER


def _set_signal_handler():
    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    _GLOBAL_SIGNAL_HANDLER = dist_signal_handler.DistributedSignalHandler().__enter__()

def set_tracing_group():
    global _GLOBAL_TRACING_GROUP
    _ensure_var_is_not_initialized(_GLOBAL_TRACING_GROUP, 'tracing group')
    _GLOBAL_TRACING_GROUP=torch.distributed.new_group()

def get_tracing_group():
    global _GLOBAL_TRACING_GROUP
    _ensure_var_is_initialized(_GLOBAL_TRACING_GROUP, "tracing group")

def set_global_variables(args, build_tokenizer=True):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    set_args(args)

    _build_num_microbatches_calculator(args)
    if build_tokenizer:
        _ = _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_wandb_writer(args)
    _set_adlr_autoresume(args)
    _set_timers(args)
    _set_tracers()
    _set_flags(args)
    _set_disturbance(args)
    _set_compresser()
    if args.exit_signal_handler:
        _set_signal_handler()


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def set_retro_args(retro_args):
    global _GLOBAL_RETRO_ARGS
    _GLOBAL_RETRO_ARGS = retro_args


def _build_num_microbatches_calculator(args):

    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR,
                                   'num microbatches calculator')

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        args)


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if hasattr(args, 'tensorboard_dir') and \
       args.tensorboard_dir and args.rank == (args.world_size - 1):
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir,
                max_queue=args.tensorboard_queue_size)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


def _set_wandb_writer(args):
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER,
                                   'wandb writer')
    if getattr(args, 'wandb_project', '') and args.rank == (args.world_size - 1):
        if args.wandb_exp_name == '':
            raise ValueError("Please specify the wandb experiment name!")

        import wandb
        if args.wandb_save_dir:
            save_dir = args.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(args.save, 'wandb')
        wandb_kwargs = {
            'dir': save_dir,
            'name': args.wandb_exp_name,
            'project': args.wandb_project,
            'config': vars(args)}
        os.makedirs(wandb_kwargs['dir'], exist_ok=True)
        wandb.init(**wandb_kwargs)
        _GLOBAL_WANDB_WRITER = wandb


def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume')

    if args.adlr_autoresume:
        if args.rank == 0:
            print('enabling autoresume ...', flush=True)
        sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
        try:
            from userlib.auto_resume import AutoResume
        except BaseException:
            print('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)
    
def _set_tracers():
    """Initialize tracers."""
    global _GLOBAL_TRACERS
    _ensure_var_is_not_initialized(_GLOBAL_TRACERS, 'tracers')
    _GLOBAL_TRACERS = Tracer()

def _set_flags(args):
    """Set the flags."""
    global _GLOBAL_FLAGS
    _ensure_var_is_not_initialized(_GLOBAL_FLAGS, 'flags')
    _GLOBAL_FLAGS = Flags(args)

def _set_disturbance(args):
    global _GLOBAL_DISTURBANCE
    _ensure_var_is_not_initialized(_GLOBAL_DISTURBANCE, 'change')
    _GLOBAL_DISTURBANCE = Disturbance(args)

def _set_compresser():
    global _GLOBAL_COMPRESSER
    _GLOBAL_COMPRESSER=DefaultCompresser()

def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)

def get_report():
    return _GLOBAL_REPORT

def set_report(func):
    global _GLOBAL_REPORT
    _GLOBAL_REPORT = func

def unset_report():
    global _GLOBAL_REPORT
    _GLOBAL_REPORT = lambda name, args, tensor: None

class FlagType(Enum):
    INVALID_FLAG = 0
    QKV_mat_mul = 1
    RawAttentionScore_mat_mul = 2
    ContextLayer_mat_mul = 3
    MLP1_mat_mul = 4
    MLP2_mat_mul = 5
    Result = 6
    MLP2_Plot = 7

class DefaultCompresser:
    def __init__(self): pass
    def compress(self, name, data):
        flag_type = name[1]
        if flag_type == FlagType.QKV_mat_mul:
            n = data.shape[1]; return True, [n], data.reshape(data.shape[0], n, 96, -1).mean(dim=-1).flatten()
        elif flag_type == FlagType.RawAttentionScore_mat_mul:
            n, m = data.shape[2], data.shape[3]; return True, [n, m], data[:, 0, :, :].flatten()
        elif flag_type == FlagType.MLP1_mat_mul or flag_type == FlagType.MLP2_mat_mul or flag_type == FlagType.ContextLayer_mat_mul:
            n = data.shape[1]; return True, [n], data.reshape(data.shape[0], n, 64, -1).mean(dim=-1).flatten()
        return False, [], torch.tensor([])

class Tracer:
    def __init__(self) -> None: pass

    @staticmethod
    def report(name, tensor_data):
        device = torch.cuda.current_device()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        tensor_data_cont = tensor_data.contiguous()
        if rank == 0:
            tensor_list = [torch.zeros_like(tensor_data_cont, dtype=tensor_data_cont.dtype, device=device) for _ in range(world_size)]
        else:
            tensor_list = None
        torch.distributed.gather(tensor_data_cont, tensor_list, dst=0)

        if rank == 0:
            aggregated_tensor = None

            if name[1] == FlagType.QKV_mat_mul:
                tensor_list0, tensor_list1, tensor_list2 = [], [], []
                for id_rank in range(world_size):
                    chunks = torch.chunk(tensor_list[id_rank], 3, dim=2)
                    tensor_list0.append(chunks[0])
                    tensor_list1.append(chunks[1])
                    tensor_list2.append(chunks[2])
                tensor0 = torch.cat(tensor_list0, dim=2)
                tensor1 = torch.cat(tensor_list1, dim=2)
                tensor2 = torch.cat(tensor_list2, dim=2)
                aggregated_tensor = torch.cat([tensor0, tensor1, tensor2], dim=2)

            elif name[1] == FlagType.RawAttentionScore_mat_mul:
                aggregated_tensor = torch.cat(tensor_list, dim=1)

            elif name[1] == FlagType.MLP1_mat_mul:
                aggregated_tensor = torch.cat(tensor_list, dim=2)

            elif name[1] == FlagType.MLP2_mat_mul:
                current_token_mlp2_output = tensor_list[0]
                if mlp2_record is not None and name[0] < len(mlp2_record) and name[0] > 0 :
                    if mlp2_record[name[0]] is None:
                        mlp2_record[name[0]] = current_token_mlp2_output
                    else:
                        mlp2_record[name[0]] = torch.cat([mlp2_record[name[0]], current_token_mlp2_output.clone()], dim=1)
                aggregated_tensor = current_token_mlp2_output

            elif name[1] == FlagType.ContextLayer_mat_mul:
                aggregated_tensor = torch.cat(tensor_list, dim=2)

            else:
                return

            valid, comp_args, compressed_tensor = get_compresser().compress(name, aggregated_tensor)
            assert valid
            get_report()(name, comp_args, compressed_tensor)

    def tik_tensor(self, name, raw):
        with torch.no_grad():
            Tracer.report(name, raw)

    def tik_result(self, result_logits, sampled_token_ids):
        name = (0, FlagType.Result)
        softmaxed_probs = torch.nn.functional.softmax(result_logits, dim=-1)
        
        def formatter(id, logit, softmaxed):
            return {
                "id": id,
                "token": get_tokenizer().decoder[id],
                "logit": logit[id].item(),
                "probability": softmaxed[id].item()
            }
        
        sampled_token_info = [formatter(sampled_token_ids[batch_idx].item(), result_logits[batch_idx], softmaxed_probs[batch_idx]) for batch_idx in range(softmaxed_probs.shape[0])]

        top_k_probs_list = []
        _, indices = torch.topk(softmaxed_probs, k=20, dim=1)
        for batch_idx in range(softmaxed_probs.shape[0]):
            for token_idx in indices[batch_idx].tolist():
                top_k_probs_list.append(formatter(token_idx, result_logits[batch_idx], softmaxed_probs[batch_idx]))

        get_report()(name, sampled_token_info, top_k_probs_list)


    @staticmethod
    def tik_end():
        if torch.distributed.get_rank() == 0:
            if mlp2_record is not None:
                args = get_args()
                for i in range(1, args.num_layers + 1):
                    if i < len(mlp2_record) and mlp2_record[i] is not None:
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(mlp2_record[i].reshape(-1, mlp2_record[i].shape[-1]).cpu())
                        pca = PCA(n_components=2)
                        reduced_data = pca.fit_transform(data_scaled)
                        get_report()((i, FlagType.MLP2_Plot), [mlp2_record[i].shape[0], mlp2_record[i].shape[1]], torch.tensor(reduced_data).flatten())

class Flags:
    """Global flags to record the status of the training process"""

    def __init__(self, args):
        self.num_layers = args.num_layers
        self.flags: Dict[FlagType, Dict[int, bool]] = {
            FlagType.INVALID_FLAG: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.QKV_mat_mul: {i: True for i in range(1, self.num_layers + 1)},
            FlagType.RawAttentionScore_mat_mul: {i: True for i in range(1, self.num_layers + 1)},
            FlagType.ContextLayer_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.MLP1_mat_mul: {i: True for i in range(1, self.num_layers + 1)},
            FlagType.MLP2_mat_mul: {i: True for i in range(1, self.num_layers + 1)},
        }

    def get_flag(self, flag_type: FlagType, layer_index: int) -> bool:
        return self.flags.get(flag_type, {}).get(layer_index, False)

    def set_by_configs(self, configs: Dict[str, Any]):
        val = True if configs.get("QKV_mat_mul", "False").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.QKV_mat_mul][i] = val
        
        val = True if configs.get("RawAttentionScore_mat_mul", "False").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.RawAttentionScore_mat_mul][i] = val
        
        val = True if configs.get("ContextLayer_mat_mul", "False").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.ContextLayer_mat_mul][i] = val
        
        val = True if configs.get("MLP1_mat_mul", "True").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.MLP1_mat_mul][i] = val
        
        val = True if configs.get("MLP2_mat_mul", "True").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.MLP2_mat_mul][i] = val

def noise1_factory(coef: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * coef
    return fn

def noise2_factory(val: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(x: torch.Tensor) -> torch.Tensor:
        rand_factors = torch.rand_like(x) * (val * 2) + (1 - val)
        return x * rand_factors
    return fn

NOISE_REGISTRY: Dict[str, Callable[..., Callable]] = {
    "noise1":  noise1_factory,
    "noise2":  noise2_factory,
}

class Disturbance:
    def __init__(self, args):
        self.weight_perturbation = False
        self.weight_perturbation_fn = None

        self.calculation_perturbation = False
        self.calculation_perturbation_fn = None

        self.system_perturbation = False
        self.system_perturbation_fn = None

    def set_by_configs(self, configs: Dict[str, Any]):
        if configs.get("weight_perturbation", False):
            self.weight_perturbation = True
            self.weight_perturbation_fn = NOISE_REGISTRY[configs["weight_perturbation_fn"]](configs["weight_perturbation_coef"])
        else:
            self.weight_perturbation = False
            self.weight_perturbation_fn = None
        
        if configs.get("calculation_perturbation", False):
            self.calculation_perturbation = True
            self.calculation_perturbation_fn = NOISE_REGISTRY[configs["calculation_perturbation_fn"]](configs["calculation_perturbation_coef"])
        else:
            self.calculation_perturbation = False
            self.calculation_perturbation_fn = None
        
        if configs.get("system_perturbation", False):
            self.system_perturbation = True
            self.system_perturbation_fn = NOISE_REGISTRY[configs["system_perturbation_fn"]](configs["system_perturbation_coef"])
        else:
            self.system_perturbation = False
            self.system_perturbation_fn = None
