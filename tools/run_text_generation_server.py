# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import os
import sys
from argparse import Namespace
from contextlib import nullcontext
from typing import Union

from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
import torch

import megatron
from megatron.core.inference.engines import AbstractEngine, StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.models.gpt import GPTModel
from megatron.training import get_model
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec, get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import import_module
from megatron.inference.text_generation.mcore_engine_server import (
    ModelInferenceWrapperServer,
    run_mcore_engine,
)
from megatron.inference.text_generation_server import MegatronServer, InferenceWSServer
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from megatron.core import mpu
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron


def model_provider(
    pre_process=True, post_process=True
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the core GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """

    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')

    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=False,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
                )

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            rope_scaling_factor=args.rope_scaling_factor,
        )

    return model


def get_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Get the relevant backend for running inference

    This function will automatically choose the TRTLLMBackend when possible, and default to Mcore
    backend if the user does not specify any backends. TRTLLMBackend is not implmented yet.

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model.

    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()

    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
        inference_max_seq_length=args.inference_max_seq_length,
        inference_max_requests=args.inference_max_batch_size,
    )

    inference_wrapped_model = ModelInferenceWrapperServer(model, inference_wrapper_config)
    text_generation_controller = TextGenerationController(inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer)
    return StaticInferenceEngine(
        text_generation_controller=text_generation_controller, max_batch_size=args.max_batch_size
    )


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument(
        "--port", type=int, default=5000, help='port for text generation server to run on'
    )
    group.add_argument(
        "--enable-ws-server",
        action='store_true',
        default=False,
        help='Enable WebSocket server for text generation',
    )
    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=1, help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument(
        "--return-log-probs",
        action='store_true',
        default=True,
        help='Return the log probabilities of the final output tokens',
    )
    group.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=30,
        help='Number of tokens to generate for each prompt',
    )
    group.add_argument(
        "--prompts",
        metavar='N',
        type=str,
        nargs='+',
        help='Input prompts with each prompt within quotes and seperated by space',
    )
    group.add_argument(
        "--max-batch-size", type=int, default=8, help='Max number of prompts to process at once'
    )
    return parser


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_text_generate_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'exit_on_missing_checkpoint': True,
        },
    )

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text " "generation.")
    args.exit_on_missing_checkpoint = True

    # Set up model and load checkpoint
    load_context = nullcontext()
    if args.fp8:
        from transformer_engine.pytorch.fp8 import fp8_model_init

        load_context = fp8_model_init()
    with load_context:
        model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    model.eval()

    inference_engine = get_inference_engine(args, model)

    if args.enable_cuda_graph:
        print(f"Running warmup for CUDA graphs...")
        inference_engine.generate(
            prompts=["Test prompt"], sampling_params=SamplingParams(num_tokens_to_generate=10)
        )

    if args.enable_ws_server:
        if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            server = InferenceWSServer(inference_engine, args)
            server.run("0.0.0.0", port=args.port)

        while True:
            from megatron.core.tensor_tracer import get_tt_flags
            from megatron.core.tensor_disturbance import get_disturbance
            obj_list = [None]
            torch.distributed.broadcast_object_list(obj_list, 0)
            get_tt_flags().set_by_configs(obj_list[0])
            torch.distributed.broadcast_object_list(obj_list, 0)
            get_disturbance().set_by_configs(obj_list[0])
            choice = torch.tensor(1, dtype=torch.long, device='cuda')
            torch.distributed.broadcast(choice, 0)
            if choice.item() == 0:
                try:
                    run_mcore_engine(inference_engine)
                except ValueError as ve:
                    pass
    else:
        if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            server = MegatronServer(inference_engine, args)
            server.run("0.0.0.0", port=args.port)

        while True:
            choice = torch.tensor(1, dtype=torch.long, device='cuda')
            torch.distributed.broadcast(choice, 0)
            if choice.item() == 0:
                try:
                    run_mcore_engine(inference_engine)
                except ValueError as ve:
                    pass
