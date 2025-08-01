# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import os
import sys
import torch

from megatron.core import Timers
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator, unset_num_microbatches_calculator
from megatron.training import dist_signal_handler
from megatron.training.tokenizer import build_tokenizer
from megatron.training.trace import Tracer

_GLOBAL_ARGS = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ONE_LOGGER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_SIGNAL_HANDLER = None
_GLOBAL_TRACER = Tracer()

def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


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


def get_one_logger():
    """Return one logger. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ONE_LOGGER

def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS


def get_tracer():
    """Return tracer."""
    return _GLOBAL_TRACER


def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    return _GLOBAL_SIGNAL_HANDLER


def _set_signal_handler():
    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    _GLOBAL_SIGNAL_HANDLER = dist_signal_handler.DistributedSignalHandler().__enter__()



def set_global_variables(args, build_tokenizer=True):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    set_args(args)

    init_num_microbatches_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        args.data_parallel_size,
        args.decrease_batch_size_if_needed,
    )
    if build_tokenizer:
        _ = _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_wandb_writer(args)
    _set_one_logger(args)
    _set_adlr_autoresume(args)
    _set_timers(args)

    from megatron.core.tensor_tracer import _set_tensor_tracers, _set_tt_flags, _set_compressor
    from megatron.core.tensor_disturbance import _set_disturbance
    _set_tensor_tracers()
    _set_tt_flags(args)
    _set_disturbance(args)
    _set_compressor()

    if args.exit_signal_handler:
        _set_signal_handler()


def unset_global_variables():
    """Unset global vars.

    Useful for multiple runs. See `tests/unit_tests/ckpt_converter/test_ckpt_converter.py` for an example.
    """

    global _GLOBAL_ARGS
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    global _GLOBAL_TOKENIZER
    global _GLOBAL_TENSORBOARD_WRITER
    global _GLOBAL_WANDB_WRITER
    global _GLOBAL_ONE_LOGGER
    global _GLOBAL_ADLR_AUTORESUME
    global _GLOBAL_TIMERS
    global _GLOBAL_SIGNAL_HANDLER

    _GLOBAL_ARGS = None
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    _GLOBAL_TOKENIZER = None
    _GLOBAL_TENSORBOARD_WRITER = None
    _GLOBAL_WANDB_WRITER = None
    _GLOBAL_ONE_LOGGER = None
    _GLOBAL_ADLR_AUTORESUME = None
    _GLOBAL_TIMERS = None
    _GLOBAL_SIGNAL_HANDLER = None

    unset_num_microbatches_calculator()


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    _GLOBAL_TRACER.global_args = args
    if args.trace_interval is not None and args.continuous_trace_iterations is not None:
        _GLOBAL_TRACER.interval = args.trace_interval
        _GLOBAL_TRACER.continuous_trace_iters = args.continuous_trace_iterations
    else:
        # Provide default values if not set, to avoid runtime errors.
        _GLOBAL_TRACER.interval = 1000
        _GLOBAL_TRACER.continuous_trace_iters = 1


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


def _set_one_logger(args):
    global _GLOBAL_ONE_LOGGER
    _ensure_var_is_not_initialized(_GLOBAL_ONE_LOGGER, 'one logger')

    if args.enable_one_logger and args.rank == (args.world_size - 1):
        if args.one_logger_async or getattr(args, 'wandb_project', ''):
            one_logger_async = True
        else:
            one_logger_async = False
        try:
            from one_logger import OneLogger
            config = {
               'project': args.one_logger_project,
               'name': args.one_logger_run_name,
               'async': one_logger_async,
            }
            one_logger = OneLogger(config=config)
            _GLOBAL_ONE_LOGGER = one_logger
        except Exception:
            print('WARNING: one_logger package is required to enable e2e metrics '
                  'tracking. please go to '
                  'https://confluence.nvidia.com/display/MLWFO/Package+Repositories'
                  ' for details to install it')

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
        except ImportError:
            print('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)

def destroy_global_vars():
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None

    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None

    global _GLOBAL_TENSORBOARD_WRITER
    _GLOBAL_TENSORBOARD_WRITER = None

    global _GLOBAL_WANDB_WRITER
    _GLOBAL_WANDB_WRITER = None

    global _GLOBAL_ONE_LOGGER
    _GLOBAL_ONE_LOGGER = None

    global _GLOBAL_ADLR_AUTORESUME
    _GLOBAL_ADLR_AUTORESUME = None

    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = None

    global _GLOBAL_SIGNAL_HANDLER
    _GLOBAL_SIGNAL_HANDLER = None
