# Copyright 2025 Suanzhi Future Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import torch
import math
import megatron.virtual_tensor_parallel_communication as dist
from megatron.training import get_args, get_tokenizer
from enum import Enum
from typing import Dict, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_GLOBAL_TT_FLAGS = None
_GLOBAL_TENSOR_TRACERS = None
_GLOBAL_REPORT = lambda name, args, tensor: None
_GLOBAL_COMPRESSOR = None
mlp2_record = None

def set_filter():
    global mlp2_record
    mlp2_record = [None for i in range(get_args().num_layers + 1)]

def _set_tensor_tracers():
    global _GLOBAL_TENSOR_TRACERS
    _GLOBAL_TENSOR_TRACERS = TensorTracers()

def _set_tt_flags(args):
    global _GLOBAL_TT_FLAGS
    _GLOBAL_TT_FLAGS = TTFlags(args)

def _set_compressor():
    global _GLOBAL_COMPRESSOR
    _GLOBAL_COMPRESSOR=DefaultCompressor()

def set_report(func):
    global _GLOBAL_REPORT
    _GLOBAL_REPORT = func

def unset_report():
    global _GLOBAL_REPORT
    _GLOBAL_REPORT = lambda name, args, tensor: None

def get_tensor_tracers():
    return _GLOBAL_TENSOR_TRACERS

def get_tt_flags():
    return _GLOBAL_TT_FLAGS

def get_compressor():
    return _GLOBAL_COMPRESSOR

def get_report():
    return _GLOBAL_REPORT

class FlagType(Enum):
    INVALID_FLAG = 0
    QKV_mat_mul = 1
    RawAttentionScore_mat_mul = 2
    ContextLayer_mat_mul = 3
    MLP1_mat_mul = 4
    MLP2_mat_mul = 5
    Result = 6
    MLP2_Plot = 7

class DefaultCompressor:
    def __init__(self):
        self.configs = {
            "QKV": {
                "pixels": 96,
                "method": "data.mean(dim=-1)"
            },
            "MLP": {
                "pixels": 64,
                "method": "data.mean(dim=-1)"
            }
        }
    def set_by_configs(self, configs: Dict[str, Any]):
        self.configs = configs
    def compress_tensor(self, data_in, pixels, method):
        B, S, F = data_in.shape
        chunk_size = math.ceil(F / pixels)
        padded_len = chunk_size * pixels
        padded_data = torch.nn.functional.pad(data_in, (0, padded_len - F))
        data_for_eval = padded_data.reshape(B, S, pixels, chunk_size)
        try:
            compressed = eval(method, {}, {"data": data_for_eval})
        except Exception as e:
            print(f"Error in compressing tensor with method '{method}': {e}")
            compressed = data_for_eval.mean(dim=-1)
        return compressed
    def compress_1d_tensor(self, data_in, pixels, method):
        B, S, F = data_in.shape
        chunk_size = math.ceil(F / pixels)
        padded_len = chunk_size * pixels
        padded_data = torch.nn.functional.pad(data_in, (0, padded_len - F))
        data_for_eval = padded_data.reshape(B, S, pixels, chunk_size)
        try:
            compressed = eval(method, {}, {"data": data_for_eval}).flatten()
        except Exception as e:
            print(f"Error in compressing tensor with method '{method}': {e}")
            compressed = data_for_eval.mean(dim=-1).flatten()  # Fallback to mean if eval fails
        return compressed
    def compress(self, name, data):
        flag_type = name[1]
        if flag_type == FlagType.QKV_mat_mul:
            n = data.shape[1]; return True, [n], self.compress_1d_tensor(data, self.configs["QKV"]["pixels"], self.configs["QKV"]["method"])
        elif flag_type == FlagType.RawAttentionScore_mat_mul:
            np, n, m = data.shape[1], data.shape[2], data.shape[3]; return True, [np, n, m], data[:, :, :, :].flatten()
        elif flag_type == FlagType.MLP1_mat_mul or flag_type == FlagType.MLP2_mat_mul or flag_type == FlagType.ContextLayer_mat_mul:
            n = data.shape[1]; return True, [n], self.compress_1d_tensor(data, self.configs["MLP"]["pixels"], self.configs["MLP"]["method"])
        return False, [], torch.tensor([])


class TensorTracers:
    def __init__(self) -> None: pass

    def report(self, name, tensor_data):
        from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
        device = torch.cuda.current_device()
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()

        if name[1] == FlagType.QKV_mat_mul:
            tensor_data = get_compressor().compress_tensor(tensor_data, get_compressor().configs["QKV"]["pixels"], get_compressor().configs["QKV"]["method"])
        elif name[1] == FlagType.MLP1_mat_mul or name[1] == FlagType.MLP2_mat_mul or name[1] == FlagType.ContextLayer_mat_mul:
            tensor_data = get_compressor().compress_tensor(tensor_data, get_compressor().configs["MLP"]["pixels"], get_compressor().configs["MLP"]["method"])
        tensor_data_cont = tensor_data.contiguous()
        if rank == 0:
            tensor_list = [torch.zeros_like(tensor_data_cont, dtype=tensor_data_cont.dtype, device=device) for _ in range(world_size)]
        else:
            tensor_list = None
        dist.gather(tensor_data_cont, tensor_list, dst=0, group=get_tensor_model_parallel_group())

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
                if get_tt_flags().get_flag(FlagType.MLP2_Plot, name[0]) and mlp2_record is not None and name[0] < len(mlp2_record) and name[0] > 0 :
                    if mlp2_record[name[0]] is None:
                        mlp2_record[name[0]] = current_token_mlp2_output
                    else:
                        mlp2_record[name[0]] = torch.cat([mlp2_record[name[0]], current_token_mlp2_output.clone()], dim=1)
                aggregated_tensor = current_token_mlp2_output

            elif name[1] == FlagType.ContextLayer_mat_mul:
                aggregated_tensor = torch.cat(tensor_list, dim=2)

            else:
                return

            valid, comp_args, compressed_tensor = get_compressor().compress(name, aggregated_tensor)
            assert valid
            get_report()(name, comp_args, compressed_tensor)

    def tik_tensor(self, name, raw):
        with torch.no_grad():
            TensorTracers.report(self, name, raw.detach())

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
        if dist.get_rank() == 0:
            if mlp2_record is not None:
                args = get_args()
                for i in range(1, args.num_layers + 1):
                    if i < len(mlp2_record) and mlp2_record[i] is not None:
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(mlp2_record[i].reshape(-1, mlp2_record[i].shape[-1]).cpu().float())
                        pca = PCA(n_components=2)
                        reduced_data = pca.fit_transform(data_scaled)
                        get_report()((i, FlagType.MLP2_Plot), [mlp2_record[i].shape[0], mlp2_record[i].shape[1]], torch.tensor(reduced_data).flatten())

class TTFlags:
    """Global flags to record the intermediate results of the model."""

    def __init__(self, args):
        self.num_layers = args.num_layers
        self.flags: Dict[FlagType, Dict[int, bool]] = {
            FlagType.INVALID_FLAG: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.QKV_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.RawAttentionScore_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.ContextLayer_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.MLP1_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.MLP2_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.MLP2_Plot: {i: False for i in range(1, self.num_layers + 1)},
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

        val = True if configs.get("MLP2_Plot", "False").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.MLP2_Plot][i] = val