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
from typing import Callable, Dict, Any

_GLOBAL_DISTURBANCE = None

def get_disturbance():
    return _GLOBAL_DISTURBANCE

def _set_disturbance(args):
    global _GLOBAL_DISTURBANCE
    _GLOBAL_DISTURBANCE = Disturbance(args)

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
