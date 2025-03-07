# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import os
from typing import Optional

import torch
import inc.torch as dist

from .utils import GlobalMemoryBuffer
from megatron import get_args
from megatron.virtual_tensor_parallel_communication import get_thread_index
import megatron.core.parallel_state as mpu

# Intra-layer model parallel group that the current rank belongs to.
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP_LIST = []
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP_LIST = []
# Embedding group.
_EMBEDDING_GROUP_LIST = []
# Position embedding group.
_POSITION_EMBEDDING_GROUP_LIST = []
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP_LIST = []
_DATA_PARALLEL_GROUP_GLOO_LIST = []
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP_LIST = []

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK_LIST = []
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK_LIST = []

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_RANK_LIST = []
_MPU_PIPELINE_MODEL_PARALLEL_RANK_LIST = []

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS_LIST = []

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS_LIST = []

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS_LIST = []

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS_LIST = []

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP_LIST = []
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS_LIST = []

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP_LIST = []
_DATA_PARALLEL_GROUP_WITH_CP_GLOO_LIST = []
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP_LIST = []

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP_LIST = []

# Memory buffers to avoid dynamic memory allocation

_FORWARD_BACKWARD_PARALLEL_GROUP_LIST = []
_FORWARD_BACKWARD_PARALLEL_GLOO_LIST = []
_FORWARD_BACKWARD_GLOBAL_RANKS_LIST = []

_HALF_DATA_PARALLEL_GROUP_LIST = []
_HALF_DATA_PARALLEL_GROUP_GLOO_LIST = []
_HALF_DATA_PARALLEL_GLOBAL_RANKS_LIST = []

def initiallize_list(
    tensor_model_parallel_size: int = 1:
) -> None:


def initialize_model_parallel_list(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    rank: int,
) -> None:
    """Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    mpu.initialize_model_parallel(tensor_model_parallel_size = tensor_model_parallel_size,
                            pipeline_model_parallel_size = pipeline_model_parallel_size,
                            virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size,
                            pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank,
                            use_sharp = use_sharp,
                            context_parallel_size = context_parallel_size,
                            rank = rank)

    _PIPELINE_MODEL_PARALLEL_GROUP_LIST[get_thread_index()] = mpu.get_pipeline_model_parallel_group()
    # Model parallel group (both intra- and pipeline) that the current rank belongs to.
    _MODEL_PARALLEL_GROUP_LIST[get_thread_index()] = mpu.get_model_parallel_group()
    # Embedding group.
    _EMBEDDING_GROUP_LIST.append[get_thread_index()] = mpu.get_embedding_group()
    # Position embedding group.
    _POSITION_EMBEDDING_GROUP_LIST[get_thread_index()] = mpu.get_position_embedding_group()
    # Data parallel group that the current rank belongs to.
    _DATA_PARALLEL_GROUP_LIST[get_thread_index()] = mpu.get_data_parallel_group()
    _DATA_PARALLEL_GROUP_GLOO_LIST[get_thread_index()] = mpu.get_data_parallel_group_gloo()
    # tensor model parallel group and data parallel group combined
    # used for fp8 and moe training
    _TENSOR_AND_DATA_PARALLEL_GROUP_LIST[get_thread_index()] = mpu.get_tensor_and_data_parallel_group()

    # _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK_LIST.append(mpu._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK)
    # _PIPELINE_MODEL_PARALLEL_SPLIT_RANK_LIST.append(mpu.get_pipeline_model_parallel_split_rank())

    # These values enable us to change the mpu sizes on the fly.

    # A list of ranks that have a copy of the embedding.
    _EMBEDDING_GLOBAL_RANKS_LIST[get_thread_index()] = mpu._EMBEDDING_GLOBAL_RANKS

    # A list of ranks that have a copy of the position embedding.
    _POSITION_EMBEDDING_GLOBAL_RANKS_LIST[get_thread_index()] = mpu._POSITION_EMBEDDING_GLOBAL_RANKS

    # A list of global ranks for each pipeline group to ease calculation of the source
    # rank when broadcasting from the first or last pipeline stage.
    _PIPELINE_GLOBAL_RANKS_LIST[get_thread_index()] = mpu._PIPELINE_GLOBAL_RANKS

    # A list of global ranks for each data parallel group to ease calculation of the source
    # rank when broadcasting weights from src to all other data parallel ranks
    _DATA_PARALLEL_GLOBAL_RANKS_LIST[get_thread_index()] = mpu._DATA_PARALLEL_GLOBAL_RANKS

    # Context parallel group that the current rank belongs to
    _CONTEXT_PARALLEL_GROUP_LIST[get_thread_index()] = mpu._CONTEXT_PARALLEL_GROUP
    # A list of global ranks for each context parallel group to ease calculation of the
    # destination rank when exchanging KV/dKV between context parallel_ranks
    _CONTEXT_PARALLEL_GLOBAL_RANKS_LIST[get_thread_index()] = mpu._CONTEXT_PARALLEL_GLOBAL_RANKS

    # Data parallel group information with context parallel combined.
    _DATA_PARALLEL_GROUP_WITH_CP_LIST[get_thread_index()] = mpu._DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP_GLOO_LIST[get_thread_index()] = mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO
    _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP_LIST[get_thread_index()] = mpu._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP

    # combined parallel group of TP, DP, and CP used for fp8
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP_LIST[get_thread_index()] = mpu._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP

    # Memory buffers to avoid dynamic memory allocation

    _FORWARD_BACKWARD_PARALLEL_GROUP_LIST[get_thread_index()] = mpu.get_forward_backward_parallel_group()
    _FORWARD_BACKWARD_PARALLEL_GLOO_LIST[get_thread_index()] = mpu._FORWARD_BACKWARD_PARALLEL_GLOO
    _FORWARD_BACKWARD_GLOBAL_RANKS_LIST[get_thread_index()] = mpu._FORWARD_BACKWARD_GLOBAL_RANKS

    _HALF_DATA_PARALLEL_GROUP_LIST[get_thread_index()] = mpu._HALF_DATA_PARALLEL_GROUP
    _HALF_DATA_PARALLEL_GROUP_GLOO_LIST[get_thread_index()] = mpu._HALF_DATA_PARALLEL_GROUP_GLOO
    _HALF_DATA_PARALLEL_GLOBAL_RANKS_LIST[get_thread_index()] = mpu._HALF_DATA_PARALLEL_GLOBAL_RANKS


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP_LIST[get_thread_index()]


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP_LIST[get_thread_index()]

def get_forward_backward_parallel_group(with_context_parallel=False):
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _FORWARD_BACKWARD_PARALLEL_GROUP is not None
    ), 'pipeline_model parallel group is not initialized'
    return _FORWARD_BACKWARD_PARALLEL_GROUP_LIST[get_thread_index()]

def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_LIST[get_thread_index()]
    else:
        assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
        return _DATA_PARALLEL_GROUP_LIST[get_thread_index()]

def get_half_data_parallel_group(with_context_parallel=False):
    assert _HALF_DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
    return _HALF_DATA_PARALLEL_GROUP_LIST[get_thread_index()]

def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO_LIST[get_thread_index()]
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, 'data parallel group-gloo is not initialized'
        return _DATA_PARALLEL_GROUP_GLOO_LIST[get_thread_index()]


def get_context_parallel_group():
    """Get the context parallel group the caller rank belongs to."""
    assert _CONTEXT_PARALLEL_GROUP is not None, 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GROUP_LIST[get_thread_index()]


def get_context_parallel_global_ranks():
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    assert _CONTEXT_PARALLEL_GLOBAL_RANKS is not None, 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GLOBAL_RANKS_LIST[get_thread_index()]


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, 'embedding group is not initialized'
    return _EMBEDDING_GROUP_LIST[get_thread_index()]


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, 'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP_LIST[get_thread_index()]


def get_amax_reduction_group(with_context_parallel=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP_LIST[get_thread_index()]
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_LIST[get_thread_index()]


def get_tensor_and_data_parallel_group(with_context_parallel=False):
    """Get the tensor and data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP_LIST[get_thread_index()]
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_LIST[get_thread_index()]


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size

def set_temporary_tensor_parallel_rank(rank):
    global _TEMPORARY_TENSOR_MODEL_PARALLEL_RANK
    _TEMPORARY_TENSOR_MODEL_PARALLEL_RANK = rank

def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return dist.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return dist.get_world_size(group=get_pipeline_model_parallel_group())


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """Set pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank

def is_forward_stage(ignore_virtual=False):
    """Return True if in the first pipeline backward model-parallel stage, False otherwise."""
    return get_forward_backward_parallel_rank() % 2 == 0

def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_thread_index()


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return dist.get_rank(group=get_pipeline_model_parallel_group())

def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = (
            get_virtual_pipeline_model_parallel_world_size()
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)

def is_pipeline_backward_first_stage():
    """Return True if in the first pipeline backward model-parallel stage, False otherwise."""
    return is_pipeline_last_stage()

def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = dist.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = dist.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and is_pipeline_stage_after_split(rank + 1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = dist.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]

def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0

def get_forward_backward_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(
            group=get_forward_backward_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0

def get_forward_backward_parallel_dual_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _FORWARD_BACKWARD_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank = get_forward_backward_parallel_rank()
    return _FORWARD_BACKWARD_GLOBAL_RANKS[rank ^ 1]

def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=get_context_parallel_group())
    else:
        return 0


def get_context_parallel_rank():
    """Return my rank for the context parallel group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=get_context_parallel_group())
    else:
        return 0


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP = None
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
