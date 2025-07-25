# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

""" Common strategies. """

import logging
import os
from pathlib import Path

import torch
import megatron.virtual_tensor_parallel_communication as dist

from megatron.core.dist_checkpointing.mapping import ShardedStateDict, StateDict
from megatron.core.dist_checkpointing.strategies.base import (
    SaveCommonStrategy,
    StrategyAction,
    register_default_strategy,
)

from ..dict_utils import dict_list_map_inplace, nested_values
from ..mapping import CheckpointingException, ShardedObject, is_main_replica
from ..strategies.base import LoadCommonStrategy

COMMON_STATE_FNAME = 'common.pt'

logger = logging.getLogger(__name__)


def register_default_common_strategies():
    """Register default common strategies."""
    register_default_strategy(StrategyAction.LOAD_COMMON, 'torch', 1, TorchCommonLoadStrategy())
    register_default_strategy(
        StrategyAction.SAVE_COMMON, 'torch', 1, TorchCommonSaveStrategy('torch', 1)
    )


class TorchCommonSaveStrategy(SaveCommonStrategy):
    """Common save strategy leveraging native torch save/load."""

    def save_common(self, common_state_dict: StateDict, checkpoint_dir: Path):
        """Save common part of the state dict."""
        if dist.get_rank() == 0:
            torch.save(common_state_dict, checkpoint_dir / COMMON_STATE_FNAME)

    def save_sharded_objects(
        self, sharded_objects_state_dict: ShardedStateDict, checkpoint_dir: Path
    ):
        """Save sharded objects from the state dict."""
        for sh_obj in nested_values(sharded_objects_state_dict):
            if is_main_replica(sh_obj.replica_id):
                save_path = checkpoint_dir / f'{sh_obj.unique_key}.pt'
                os.makedirs(save_path.parent, exist_ok=True)
                torch.save(sh_obj.data, save_path)

    def can_handle_sharded_objects(self):
        """This strategy can handle ShardedObjects."""
        return True


class TorchCommonLoadStrategy(LoadCommonStrategy):
    """Common load strategy leveraging native torch save/load."""

    def load_common(self, checkpoint_dir: Path):
        """Load common (non-sharded) objects state dict from the checkpoint.

        Args:
            checkpoint_dir (Path): checkpoint directory

        Returns:
            StateDict: state dict with non-sharded objects from the checkpoint
        """
        load_path = Path(checkpoint_dir) / COMMON_STATE_FNAME
        try:
            return torch.load(load_path, map_location='cpu', weights_only=False)
        except FileNotFoundError as e:
            err_msg = f'Common file {load_path} does not exist'
            ckpt_files = [f.name for f in checkpoint_dir.iterdir()]
            logger.debug(f'{err_msg}. Checkpoint directory content: {ckpt_files}')
            raise CheckpointingException(err_msg) from e

    def load_sharded_objects(
        self, sharded_objects_state_dict: ShardedStateDict, checkpoint_dir: Path
    ):
        """Replaces all ShardedObject from a given state dict with values loaded from the
        checkpoint.

        Args:
            sharded_objects_state_dict (ShardedStateDict):
                sharded state dict defining what objects should be loaded.
            checkpoint_dir (Path): checkpoint directory

        Returns:
            None: sharded state dict is modified in place
        """

        def load_sharded_object(sh_obj: ShardedObject):
            sh_obj.data = None
            load_path = checkpoint_dir / f'{sh_obj.unique_key}.pt'
            try:
                loaded_obj = torch.load(load_path, weights_only=False)
            except FileNotFoundError as e:
                # Backward compatible logic: previously the save format was incorrect
                old_load_path = (checkpoint_dir / sh_obj.unique_key).with_suffix('.pt')
                try:
                    loaded_obj = torch.load(old_load_path, weights_only=False)
                except FileNotFoundError:
                    err_msg = f'Object shard {load_path} not found'
                    obj_subdir = checkpoint_dir / sh_obj.key
                    if obj_subdir.exists():
                        obj_files = [f.name for f in obj_subdir.iterdir()]
                        logger.debug(
                            f'{err_msg}. Object {sh_obj.key} directory content: {obj_files}'
                        )
                    else:
                        ckpt_files = [f.name for f in checkpoint_dir.iterdir()]
                        logger.debug(
                            f'{err_msg}. Object {sh_obj.key} directory does not exist. Checkpoint'
                            f' directory content: {ckpt_files}'
                        )
                    raise CheckpointingException(err_msg) from e
            return loaded_obj

        return dict_list_map_inplace(load_sharded_object, sharded_objects_state_dict)

    def load_sharded_metadata(self, checkpoint_dir: Path) -> ShardedStateDict:
        sharded_metadata = {}
        for subdir in checkpoint_dir.iterdir():
            if not subdir.is_dir():
                continue
            shard_files = list(subdir.glob('shard_*.pt'))
            if not shard_files:
                continue
            sh_objs = []
            for shard_file in shard_files:
                full_key = f'{subdir.name}/{shard_file.stem}'
                sh_objs.append(ShardedObject.empty_from_unique_key(full_key))

            # This is a backward-compatibility fix, where the last global shape is missing in the
            # name
            if sh_objs[0].global_shape[-1] < 0:
                max_last_offset = max(map(lambda sh_obj: sh_obj.global_offset[-1], sh_objs))
                for sh_obj in sh_objs:
                    sh_obj.global_shape = (*sh_obj.global_shape[:-1], max_last_offset + 1)

            # Update the sharded state dict
            for sh_obj in sh_objs:
                sharded_metadata[sh_obj.unique_key] = sh_obj
        return sharded_metadata

    @property
    def can_handle_sharded_objects(self):
        """This strategy can handle ShardedObjects."""
        return True

    def check_backend_compatibility(self, loaded_version):
        pass

    def check_version_compatibility(self, loaded_version):
        pass
