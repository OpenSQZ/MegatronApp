# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import megatron.virtual_tensor_parallel_communication as dist
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_with_cos_sin,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import deprecate_inference_params, divide

from .enums import AttnMaskType
from .transformer_config import TransformerConfig

try:
    from einops import rearrange
except ImportError:
    rearrange = None


try:
    from nvidia_chunked_flash_attn.flash_attn_interface import (
        flash_attn_varlen_func as flash_decode_and_prefill_kernel,
    )
except ImportError:
    flash_decode_and_prefill_kernel = None


try:
    from flash_attn import flash_attn_with_kvcache
except:
    flash_attn_with_kvcache = None


try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


@dataclass
class SelfAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a self-attention.
    """

    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


@dataclass
class CrossAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a cross-attention.
    """

    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


class Attention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        # To support both CUDA Graphs and key value with different hidden size
        self.key_hidden_size = self.hidden_size_per_attention_head
        self.val_hidden_size = self.hidden_size_per_attention_head

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=cp_comm_type,
            softmax_scale=self.config.softmax_scale,
        )

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
            and "core_attn" in self.config.recompute_modules
        )

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

    def _checkpointed_attention_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            attention_mask = inputs[3]
            attn_mask_type = inputs[5]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            output_ = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            return output_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
        hidden_states = tensor_parallel.checkpoint(
            custom_forward, False, query, key, value, attention_mask, rotary_pos_emb, attn_mask_type
        )

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_length, batch_size, dim, dtype):
        """Allocate memory to store kv cache during inference."""

        return torch.empty(
            inference_max_sequence_length,
            batch_size,
            self.num_query_groups_per_partition,
            dim,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def _adjust_key_value_for_inference(
        self,
        inference_context: BaseInferenceContext,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        rotary_pos_emb: Tensor,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Saves the generated key and value tensors to the end of the buffers in inference_context.
        Returns the full size keys and values from the provided inference_context, as well as
        adjusted rotary_pos_emb.

        Args:
            query (Tensor): Query tensor.
            key (Tensor): Key tensor.
            value (Tensor): Value tensor.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            Tuple of: query, key, value, rotary_pos_emb, attn_mask_type.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        attn_mask_type = self.attn_mask_type
        if inference_context is None:
            return query, key, value, rotary_pos_emb, attn_mask_type

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if inference_context.is_static_batching():
            if self.layer_number not in inference_context.key_value_memory_dict:
                inf_max_seq_length = inference_context.max_sequence_length
                inf_max_batch_size = inference_context.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_length, inf_max_batch_size, self.key_hidden_size, key.dtype
                )
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_length, inf_max_batch_size, self.val_hidden_size, value.dtype
                )
                inference_context.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory,
                    inference_value_memory,
                )
            else:
                # Get the pre-allocated buffers for this layer
                inference_key_memory, inference_value_memory = (
                    inference_context.key_value_memory_dict[self.layer_number]
                )

        if not inference_context.is_static_batching() or inference_context.sequence_len_offset > 0:
            # This should mean that we are past the prompt forward_step
            # and so we need to turn off masking
            attn_mask_type = AttnMaskType.no_mask

        if inference_context.is_static_batching():
            batch_start = inference_context.batch_size_offset
            batch_end = batch_start + key.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_context.sequence_len_offset
            sequence_end = sequence_start + key.size(0)
            assert sequence_end <= inference_key_memory.size(0), (
                "Current sequence length is longer than expected maximum sequence length! "
                "Increase inference_max_seq_length."
            )

        if self.config.flash_decode:
            rotary_pos_cos_q = None
            rotary_pos_sin_q = None
            rotary_pos_cos_k = None
            rotary_pos_sin_k = None

            assert inference_context.is_static_batching()
            if (
                inference_context.is_decode_only() and rotary_pos_cos is not None
            ):  # Decode phase, not prefill
                rotary_pos_cos_q = rotary_pos_cos[sequence_end - 1 : sequence_end]
                rotary_pos_sin_q = rotary_pos_sin[sequence_end - 1 : sequence_end]
                rotary_pos_cos_k = rotary_pos_cos[sequence_end - 1 : sequence_end]
                rotary_pos_sin_k = rotary_pos_sin[sequence_end - 1 : sequence_end]
            elif rotary_pos_cos is not None:  # Prefill
                rotary_pos_cos_q = rotary_pos_cos[:sequence_end]
                rotary_pos_sin_q = rotary_pos_sin[:sequence_end]
                rotary_pos_cos_k = rotary_pos_cos[:sequence_end]
                rotary_pos_sin_k = rotary_pos_sin[:sequence_end]

            # Flash Decoding assumes that the keys stored in the KV Cache already have RoPE applied.
            # Apply RoPE before we store the keys to make it compatible with flash decoding kernel
            if rotary_pos_sin_q is not None and rotary_pos_sin_k is not None:
                key = apply_rotary_pos_emb_with_cos_sin(key, rotary_pos_cos_k, rotary_pos_sin_k)
                query = apply_rotary_pos_emb_with_cos_sin(query, rotary_pos_cos_q, rotary_pos_sin_q)
        else:
            rotary_pos_cos_q = None
            rotary_pos_sin_q = None

        # Adjust rotary embeddings.
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if inference_context.is_static_batching():
                q_pos_emb = q_pos_emb[sequence_start:sequence_end, :, :, :]
                k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            else:
                pass
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        if inference_context.is_static_batching():
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key
            inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value
            key = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
            value = inference_value_memory[:sequence_end, batch_start:batch_end, ...]
        else:
            # Apply rotary embeddings before appending KV cache.
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                key = inference_context.apply_rotary_emb_key(key, k_pos_emb, self.config)
                rotary_pos_emb = (q_pos_emb, None)  # key rotary emb has been applied

            # Append key/value data tensors to cache.
            inference_context.append_key_value_cache(self.layer_number, key, value)

            # Read key/value *pointer* tensors from cache.
            key, value = inference_context.key_value_cache(self.layer_number)

        return query, key, value, rotary_pos_emb, attn_mask_type

    @abstractmethod
    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    def flash_decode(
        self,
        sequence_len_offset: Tensor,
        query_layer: Tensor,
        key_layer: Tensor,
        value_layer: Tensor,
        inference_key_memory: Tensor,
        inference_value_memory: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
    ) -> (Tensor, Tensor):
        """
        The flash decoding kernel will do the following in a single execution:
        1. Compute RoPE embedding with precomputed cos & sin tensors
        2. Update the KV Cache
        3. Performs the flash attention operation
        """
        assert flash_attn_with_kvcache is not None, (
            "Flash Decoding requires the flash_attn_with_kvcache kernel, "
            "available in the flash-attn package."
        )
        q = query_layer.permute(1, 0, 2, 3)
        k = key_layer.permute(1, 0, 2, 3)
        v = value_layer.permute(1, 0, 2, 3)
        k_cache = inference_key_memory.permute(1, 0, 2, 3)
        v_cache = inference_value_memory.permute(1, 0, 2, 3)

        if rotary_cos is not None:
            rotary_cos = rotary_cos.to(query_layer.dtype)
        if rotary_sin is not None:
            rotary_sin = rotary_sin.to(query_layer.dtype)

        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=sequence_len_offset,
            rotary_interleaved=False,
        )
        return out

    def flash_decode_and_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        seqlen_q: Optional[int] = None,
        seqlen_k: Optional[int] = None,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
    ) -> Tensor:
        """Flash attention kernel for mixed decode and prefill samples.

        Args:
            q (Tensor): Query tensor.
            k (Tensor): Key tensor.
            v (Tensor): Value tensor.
            seqlen_q (Optional[int]): Query total sequence length.
            seqlen_k (Optional[int]): Key total sequence length.
            cu_seqlens_q (Optional[Tensor]): Cumulative query sequence lengths.
            cu_seqlens_k (Optional[Tensor]): Cumulative key sequence lengths.

        Return:
            (Tensor) Attention output.
        """

        assert not self.training

        # Default variables.
        if seqlen_q is None:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
        else:
            batch_size = 1
        if seqlen_k is None:
            seqlen_k = k.shape[1]

        if cu_seqlens_q is None:
            cu_seqlens_q = torch.arange(
                0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device
            )

        # turn off FA causal mask after first inference autoregressive iteration
        # only on first autoregressive step q,k,v have same seqlen
        # TODO: pass is_causal per sample to flash attentation
        if cu_seqlens_k is None:
            cu_seqlens_k = torch.arange(
                0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=q.device
            )

        # Contiguous tensors.
        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Flash attn kernel.
        output_total = flash_decode_and_prefill_kernel(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            dropout_p=0,
            softmax_scale=None,
            causal=True,
            num_heads_k=self.config.num_query_groups,
        )
        output_total = rearrange(output_total, '(b s) ... -> b s ...', b=batch_size)

        return output_total

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """

        from megatron.training.global_vars import get_args, get_tracer
        args = get_args()
        if args.trace:
            tracer = get_tracer()
            with tracer.scope("attention"):
                inference_context = deprecate_inference_params(inference_context, inference_params)

                if inference_context and inference_context.is_dynamic_batching():
                    assert (
                        flash_decode_and_prefill_kernel is not None
                    ), "Internal use only: install package `nvidia_chunked_flash_attn`."

                # hidden_states: [sq, b, h]
                if self.config.flash_decode and not self.training and inference_context is not None:
                    rotary_pos_emb = None
                else:
                    assert rotary_pos_cos is None and rotary_pos_sin is None

                # For self attention we just duplicate the rotary_pos_emb if it isn't already
                if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
                    rotary_pos_emb = (rotary_pos_emb,) * 2

                # =====================
                # Query, Key, and Value
                # =====================
                # Get the query, key and value tensors based on the type of attention -
                # self or cross attn.
                query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

                # ===================================================
                # Adjust key, value, and rotary_pos_emb for inference
                # ===================================================

                # This branch only runs in the decode phase of flash decoding and returns after the linear
                # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
                if (
                    self.config.flash_decode
                    and inference_context is not None
                    and inference_context.is_decode_only()
                    and not self.training
                    and rotary_pos_cos is not None
                ):
                    assert self.layer_number in inference_context.key_value_memory_dict
                    assert inference_context.sequence_len_offset is not None
                    inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
                        self.layer_number
                    ]
                    output = self.flash_decode(
                        sequence_len_offset=sequence_len_offset,
                        query_layer=query,
                        key_layer=key,
                        value_layer=value,
                        inference_key_memory=inference_key_memory,
                        inference_value_memory=inference_value_memory,
                        rotary_cos=rotary_pos_cos,
                        rotary_sin=rotary_pos_sin,
                    )
                    out = output.transpose(0, 1).contiguous()
                    context_layer = out.view(out.size(0), out.size(1), -1)
                    output, bias = self.linear_proj(context_layer)
                    return output, bias

                query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
                    inference_context,
                    query,
                    key,
                    value,
                    rotary_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    sequence_len_offset,
                )

                if packed_seq_params is not None:
                    query = query.squeeze(1)
                    key = key.squeeze(1)
                    value = value.squeeze(1)

                # ================================================
                # relative positional embedding (rotary embedding)
                # ================================================
                if rotary_pos_emb is not None and not self.config.flash_decode:
                    q_pos_emb, k_pos_emb = rotary_pos_emb

                    if packed_seq_params is not None:
                        if packed_seq_params.cu_seqlens_q_padded is not None:
                            cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                        else:
                            cu_seqlens_q = packed_seq_params.cu_seqlens_q
                        if packed_seq_params.cu_seqlens_kv_padded is not None:
                            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                        else:
                            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
                    else:
                        cu_seqlens_q = cu_seqlens_kv = None

                    if q_pos_emb is not None:
                        # TODO VIJAY: simplify
                        if inference_context is None or inference_context.is_static_batching():
                            query = apply_rotary_pos_emb(
                                query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
                            )
                        else:
                            query = inference_context.apply_rotary_emb_query(
                                query, q_pos_emb, self.config, cu_seqlens_q
                            )
                    if k_pos_emb is not None:
                        key = apply_rotary_pos_emb(
                            key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv
                        )

                    # TODO, can apply positional embedding to value_layer so it has
                    # absolute positional embedding.
                    # otherwise, only relative positional embedding takes effect
                    # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

                # ==================================
                # core attention computation
                # ==================================

                if self.checkpoint_core_attention and self.training:
                    core_attn_out = self._checkpointed_attention_forward(
                        query,
                        key,
                        value,
                        attention_mask,
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params,
                    )
                else:
                    if inference_context is None or inference_context.is_static_batching():
                        # Static batching attention kernel.
                        core_attn_out = self.core_attention(
                            query,
                            key,
                            value,
                            attention_mask,
                            attn_mask_type=attn_mask_type,
                            attention_bias=attention_bias,
                            packed_seq_params=packed_seq_params,
                        )

                    else:
                        # Dynamic batching attention kernel.
                        q, k, v = (query, key, value)
                        cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                        cu_kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                        core_attn_out = self.flash_decode_and_prefill(
                            q, k, v, max_seqlen_q, max_seqlen_k, cu_query_lengths, cu_kv_lengths
                        )
                        core_attn_out = core_attn_out.squeeze(0).unsqueeze(1)
                        core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

                if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
                    # reshape to same output shape as unpacked case
                    # (t, np, hn) -> (t, b=1, h=np*hn)
                    # t is the pack size = sum (sq_i)
                    # note that batch is a dummy dimension in the packed case
                    core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

                # =================
                # Output. [sq, b, h]
                # =================

                output, bias = self.linear_proj(core_attn_out)

                return output, bias
        else:
            inference_context = deprecate_inference_params(inference_context, inference_params)

            if inference_context and inference_context.is_dynamic_batching():
                assert (
                    flash_decode_and_prefill_kernel is not None
                ), "Internal use only: install package `nvidia_chunked_flash_attn`."

            # hidden_states: [sq, b, h]
            if self.config.flash_decode and not self.training and inference_context is not None:
                rotary_pos_emb = None
            else:
                assert rotary_pos_cos is None and rotary_pos_sin is None

            # For self attention we just duplicate the rotary_pos_emb if it isn't already
            if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb,) * 2

            # =====================
            # Query, Key, and Value
            # =====================
            # Get the query, key and value tensors based on the type of attention -
            # self or cross attn.
            query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

            # ===================================================
            # Adjust key, value, and rotary_pos_emb for inference
            # ===================================================

            # This branch only runs in the decode phase of flash decoding and returns after the linear
            # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
            if (
                self.config.flash_decode
                and inference_context is not None
                and inference_context.is_decode_only()
                and not self.training
                and rotary_pos_cos is not None
            ):
                assert self.layer_number in inference_context.key_value_memory_dict
                assert inference_context.sequence_len_offset is not None
                inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
                    self.layer_number
                ]
                output = self.flash_decode(
                    sequence_len_offset=sequence_len_offset,
                    query_layer=query,
                    key_layer=key,
                    value_layer=value,
                    inference_key_memory=inference_key_memory,
                    inference_value_memory=inference_value_memory,
                    rotary_cos=rotary_pos_cos,
                    rotary_sin=rotary_pos_sin,
                )
                out = output.transpose(0, 1).contiguous()
                context_layer = out.view(out.size(0), out.size(1), -1)
                output, bias = self.linear_proj(context_layer)
                return output, bias

            query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
                inference_context,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            )

            if packed_seq_params is not None:
                query = query.squeeze(1)
                key = key.squeeze(1)
                value = value.squeeze(1)

            # ================================================
            # relative positional embedding (rotary embedding)
            # ================================================
            if rotary_pos_emb is not None and not self.config.flash_decode:
                q_pos_emb, k_pos_emb = rotary_pos_emb

                if packed_seq_params is not None:
                    if packed_seq_params.cu_seqlens_q_padded is not None:
                        cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                    else:
                        cu_seqlens_q = packed_seq_params.cu_seqlens_q
                    if packed_seq_params.cu_seqlens_kv_padded is not None:
                        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                    else:
                        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
                else:
                    cu_seqlens_q = cu_seqlens_kv = None

                if q_pos_emb is not None:
                    # TODO VIJAY: simplify
                    if inference_context is None or inference_context.is_static_batching():
                        query = apply_rotary_pos_emb(
                            query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
                        )
                    else:
                        query = inference_context.apply_rotary_emb_query(
                            query, q_pos_emb, self.config, cu_seqlens_q
                        )
                if k_pos_emb is not None:
                    key = apply_rotary_pos_emb(
                        key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv
                    )

                # TODO, can apply positional embedding to value_layer so it has
                # absolute positional embedding.
                # otherwise, only relative positional embedding takes effect
                # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

            # ==================================
            # core attention computation
            # ==================================

            if self.checkpoint_core_attention and self.training:
                core_attn_out = self._checkpointed_attention_forward(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
            else:
                if inference_context is None or inference_context.is_static_batching():
                    # Static batching attention kernel.
                    core_attn_out = self.core_attention(
                        query,
                        key,
                        value,
                        attention_mask,
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params,
                    )

                else:
                    # Dynamic batching attention kernel.
                    q, k, v = (query, key, value)
                    cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                    cu_kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                    core_attn_out = self.flash_decode_and_prefill(
                        q, k, v, max_seqlen_q, max_seqlen_k, cu_query_lengths, cu_kv_lengths
                    )
                    core_attn_out = core_attn_out.squeeze(0).unsqueeze(1)
                    core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

            if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
                # reshape to same output shape as unpacked case
                # (t, np, hn) -> (t, b=1, h=np*hn)
                # t is the pack size = sum (sq_i)
                # note that batch is a dummy dimension in the packed case
                core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

            # =================
            # Output. [sq, b, h]
            # =================

            output, bias = self.linear_proj(core_attn_out)

            return output, bias


class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def run_realtime_tests(self):
        """Performs a consistency check.

        This function makes sure that tensors across devices are the same during an experiment.
        This is often not guaranteed to be so because of silent hardware failures (eg, memory
        corruption loading a checkpoint, network traffic corruption encountered during
        data transmission).

        (TODO) In the future, more tensors should be checked across the training run and
        checked every X iterations. This is left for future work. Equality of tensors is probably
        not required; transmitting hashes is sufficient."""

        if not self.config.qk_layernorm:
            return

        # check that all tensor parallel and data parallel ranks have the same
        # Q & K layernorm parameters.
        rank = get_data_parallel_rank()
        inputs = torch.stack(
            [
                self.q_layernorm.weight.data,
                self.q_layernorm.bias.data,
                self.k_layernorm.weight.data,
                self.k_layernorm.bias.data,
            ]
        )
        dp_list = [torch.empty_like(inputs) for _ in range(get_data_parallel_world_size())]
        dp_list[rank] = inputs
        dist.all_gather(dp_list, inputs, group=get_data_parallel_group())

        def _compare(srcs, tgts, names, parallelism):
            assert len(srcs) == len(tgts) == len(names)
            for src, tgt, name in zip(srcs, tgts, names):
                assert torch.all(src == tgt), (
                    f"Discrepancy between {name} in {parallelism} ranks {i} and {rank}. "
                    f"Diff: {torch.norm(src - tgt)}"
                )

        for i, dp in enumerate(dp_list):
            q_w, q_b, k_w, k_b = torch.unbind(dp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "DP",
            )

        rank = get_tensor_model_parallel_rank()
        tp_list = [torch.empty_like(inputs) for _ in range(get_tensor_model_parallel_world_size())]
        tp_list[rank] = inputs
        dist.all_gather(tp_list, inputs, group=get_tensor_model_parallel_group())

        for i, tp in enumerate(tp_list):
            q_w, q_b, k_w, k_b = torch.unbind(tp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "TP",
            )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        from megatron.core.tensor_tracer import get_tt_flags, FlagType, get_tensor_tracers
        if get_tt_flags().get_flag(FlagType.QKV_mat_mul, self.layer_number):
            get_tensor_tracers().tik_tensor((self.layer_number, FlagType.QKV_mat_mul), mixed_qkv.transpose(0, 1))

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value


class CrossAttention(Attention):
    """Cross-attention layer class

    Cross-attention layer takes input with size [s, b, h] and context with size
    [s, b, h] and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
            cp_comm_type=cp_comm_type,
        )

        if self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError("Group query attention is not currently supported in cross attention.")
        assert self.query_projection_size == self.kv_projection_size

        self.linear_q = build_module(
            submodules.linear_q,
            self.config.hidden_size,
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv = build_module(
            submodules.linear_kv,
            self.config.hidden_size,
            2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv, _ = self.linear_kv(key_value_states)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv.size()[:-1] + (
            self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query, _ = self.linear_q(hidden_states)

        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        return query, key, value
