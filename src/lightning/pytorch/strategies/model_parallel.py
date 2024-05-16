# Copyright The Lightning AI team.
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
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import shutil
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Mapping, Optional, Set, Type, Union

import torch
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.strategies.fsdp import (
    _METADATA_FILENAME,
    _activation_checkpointing_kwargs,
    _auto_wrap_policy_kwargs,
    _distributed_checkpoint_load,
    _distributed_checkpoint_save,
    _get_full_state_dict_context,
    _get_sharded_state_dict_context,
    _init_cpu_offload,
    _init_sharding_strategy,
    _is_full_checkpoint,
    _is_sharded_checkpoint,
    _move_torchmetrics_to_device,
    _optimizer_has_flat_params,
    _setup_activation_checkpointing,
)
from lightning.fabric.strategies.model_parallel import _load_raw_module_state, _load_checkpoint, _setup_device_mesh
from lightning.fabric.utilities.distributed import (
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1, _TORCH_GREATER_EQUAL_2_3
from lightning.fabric.utilities.init import _EmptyInit, _has_meta_device_parameters_or_buffers, \
    _materialize_distributed_module
from lightning.fabric.utilities.load import _lazy_load, _materialize_tensors
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH, ReduceOp
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.plugins.precision import Precision
from lightning.pytorch.plugins.precision.fsdp import FSDPPrecision
from lightning.pytorch.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.strategies.strategy import TBroadcast
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class ModelParallelStrategy(ParallelStrategy):
    """Enables user-defined parallelism applied to a model.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Currently supports up to 2D parallelism. Specifically, it supports the combination of
    Fully Sharded Data-Parallel 2 (FSDP2) with Tensor Parallelism (DTensor). These PyTorch APIs are currently still
    experimental in PyTorch. Requires PyTorch 2.3 or newer.

    Arguments:
        data_parallel_size: The number of devices within a data-parallel group. Defaults to ``"auto"``, which
            sets this size to the number of nodes in the cluster.
        tensor_parallel_size: The number of devices within a tensor-parallel group. Defaults to ``"auto"``, which
            sets this size to the number of GPUs in a single node.
        save_distributed_checkpoint: If ``True``, each rank saves its shard of weights and optimizer states to a file.
            The checkpoint is a folder with as many files as the world size.
            If ``False``, the full weights and optimizer states get assembled on rank 0 and saved to a single file.

    """

    def __init__(
        self,
        data_parallel_size: Union[Literal["auto"], int] = "auto",
        tensor_parallel_size: Union[Literal["auto"], int] = "auto",
        save_distributed_checkpoint: bool = True,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
    ) -> None:
        super().__init__()
        if not _TORCH_GREATER_EQUAL_2_3:
            raise ImportError(f"{type(self).__name__} requires PyTorch 2.3 or higher.")
        self._data_parallel_size = data_parallel_size
        self._tensor_parallel_size = tensor_parallel_size
        self._save_distributed_checkpoint = save_distributed_checkpoint
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._device_mesh: Optional["DeviceMesh"] = None
        self.num_nodes = 1

    @property
    def device_mesh(self) -> "DeviceMesh":
        if self._device_mesh is None:
            raise RuntimeError("Accessing the device mesh before processes have initialized is not allowed.")
        return self._device_mesh

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        raise NotImplementedError(f"The `{type(self).__name__}` does not use the `CheckpointIO` plugin interface.")

    @checkpoint_io.setter
    @override
    def checkpoint_io(self, io: CheckpointIO) -> None:
        raise NotImplementedError(f"The `{type(self).__name__}` does not support setting a `CheckpointIO` plugin.")

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    @override
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        assert self.device_mesh is not None
        data_parallel_mesh = self.device_mesh["data_parallel"]
        return {"num_replicas": data_parallel_mesh.size(), "rank": data_parallel_mesh.get_local_rank()}

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @property
    @override
    def restore_checkpoint_after_setup(self) -> bool:
        return True

    @property
    @override
    def lightning_restore_optimizer(self) -> bool:
        return False

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self._setup_distributed()
        if self._data_parallel_size == "auto":
            self._data_parallel_size = self.num_nodes
        if self._tensor_parallel_size == "auto":
            self._tensor_parallel_size = self.num_processes
        self._device_mesh = _setup_device_mesh(
            self._data_parallel_size, self._tensor_parallel_size, self.world_size, self.root_device
        )
        # Users can access device mesh in `LightningModule.configure_model()`
        self.lightning_module._device_mesh = self._device_mesh

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        # TODO: assert that the configure_model() hook was implemented and model has
        # distributed modules

        # TODO: needed?
        # we set the device so that optimizers can be created with distributed comms.
        assert self.lightning_module is not None
        self.lightning_module._device = self.root_device

        _materialize_distributed_module(self.model, self.root_device)

        self.model = self.precision_plugin.convert_module(self.model)
        self.barrier()

        if trainer.state.fn == TrainerFn.FITTING:
            self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        if trainer.state.fn == TrainerFn.FITTING:
            _optimizers_to_device(self.optimizers, self.root_device)

    @override
    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        # If we're setting up for evaluation after fitting, we need to discard the optimizers
        # since we're rewrapping the model, otherwise optimizer param references are no longer valid
        # and subsequent checkpoint saving can fail
        self._reset_optimizers_and_schedulers()

        # TODO:
        # if self.kwargs.get("use_orig_params"):
        #     return super().setup_optimizers(trainer)

        invalid_params_error = False
        # try:
        #     # In PyTorch < 2.0, or if `use_orig_params=False` the user needs to do access
        #     # `self.trainer.model.parameters()` in configure_optimizers()
        return super().setup_optimizers(trainer)
        # except ValueError as ex:
        #     if "optimizer got an empty parameter list" not in str(ex):
        #         raise
        #     invalid_params_error = True

        # if invalid_params_error or any(not _optimizer_has_flat_params(optimizer) for optimizer in self.optimizers):
        #     # We avoid this limitation in PyTorch >= 2.0 by setting `use_orig_params=True`
            
        #     print(invalid_params_error,  any(not _optimizer_has_flat_params(optimizer) for optimizer in self.optimizers))
            
        #     raise ValueError(
        #         "The optimizer does not seem to reference any FSDP parameters. HINT: Make sure to create the"
        #         " optimizer after setting up the model by referencing `self.trainer.model.parameters()` in the"
        #         " `configure_optimizers()` hook."
        #     )
        return None

    @override
    def model_to_device(self) -> None:
        pass

    @contextmanager
    @override
    def tensor_init_context(self, empty_init: Optional[bool] = None) -> Generator[None, None, None]:
        # Materializaton happens in `_setup_module`
        empty_init_context = torch.device("meta") if empty_init else nullcontext()
        with empty_init_context, self.precision_plugin.tensor_init_context():
            yield

    @override
    def barrier(self, name: Optional[str] = None) -> None:
        if not _distributed_is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self._determine_device_ids())
        else:
            torch.distributed.barrier()

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    @override
    def reduce(
        self,
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Tensor:
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def _determine_device_ids(self) -> List[int]:
        return [self.root_device.index]

    @override
    def teardown(self) -> None:
        assert self.cluster_environment is not None
        assert self.accelerator is not None
        self.cluster_environment.teardown()
        self.precision_plugin.teardown()
        self.accelerator.teardown()

    @override
    def lightning_module_state_dict(self) -> Dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        state_dict_options = StateDictOptions(full_state_dict=(not self._save_distributed_checkpoint), cpu_offload=True)
        assert self.model is not None
        return get_model_state_dict(self.model, options=state_dict_options)

    @override
    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        # Override to do nothing, the strategy already loaded the states in `load_checkpoint()`
        pass

    @override
    def optimizer_state(self, optimizer: Optimizer) -> Dict[str, Tensor]:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import OptimStateKeyType
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

        state_dict_options = StateDictOptions(full_state_dict=(not self._save_distributed_checkpoint), cpu_offload=True)
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer

        assert self.model is not None
        state_dict = get_optimizer_state_dict(self.model, optimizer, options=state_dict_options)
        if not self._save_distributed_checkpoint:
            # Store the optimizer state dict in standard format
            state_dict = FSDP.rekey_optim_state_dict(state_dict, OptimStateKeyType.PARAM_ID, self.model)
        return state_dict

    @override
    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # Override to do nothing, the strategy already loaded the states in `load_checkpoint()`
        pass

    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                f"`{type(self).__name__}.save_checkpoint(..., storage_options=...)` is not supported because"
                f" `{type(self).__name__}` does not use the `CheckpointIO`."
            )

        path = Path(self.broadcast(filepath))
        if path.is_dir() and not self._save_distributed_checkpoint and not _is_sharded_checkpoint(path):
            raise IsADirectoryError(f"The checkpoint path exists and is a directory: {path}")

        if self._save_distributed_checkpoint:
            if path.is_file():
                path.unlink()
            path.mkdir(parents=True, exist_ok=True)

            converted_state = {"model": checkpoint.pop("state_dict")}
            converted_state.update({
                f"optimizer_{idx}": optim_state
                for idx, optim_state in enumerate(checkpoint.pop("optimizer_states", []))
            })

            _distributed_checkpoint_save(converted_state, path)

            if self.global_rank == 0:
                torch.save(checkpoint, path / _METADATA_FILENAME)
        else:
            if _is_sharded_checkpoint(path):
                shutil.rmtree(path)
            return super().save_checkpoint(checkpoint=checkpoint, filepath=path)

    @override
    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(checkpoint_path))

        checkpoint = torch.load(checkpoint_path, mmap=True, map_location="cpu")

        state = {}  # ?
        checkpoint = _load_checkpoint(path=path, state=state, strict=self.lightning_module.strict_loading)
        return checkpoint

        # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        #
        # assert self.model is not None
        # assert self.lightning_module is not None
        #
        # if _is_sharded_checkpoint(path):
        #     from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
        #
        #     state_dict_ctx = _get_sharded_state_dict_context(self.model)
        #
        #     with state_dict_ctx:
        #         module_state = {"model": self.model.state_dict()}
        #         _distributed_checkpoint_load(module_state, path)
        #         self.model.load_state_dict(module_state["model"], strict=self.lightning_module.strict_loading)
        #
        #         if self.lightning_module.trainer.state.fn == TrainerFn.FITTING and self.optimizers:
        #             from torch.distributed.checkpoint import FileSystemReader
        #
        #             # TODO: replace with newer APIs
        #             # https://github.com/pytorch/pytorch/issues/119800#issuecomment-1942156271
        #             reader = FileSystemReader(path=path)
        #             # the optimizer states must be loaded separately
        #             for idx, optim in enumerate(self.optimizers):
        #                 optim_key = f"optimizer_{idx}"
        #                 optim_state = load_sharded_optimizer_state_dict(
        #                     model_state_dict=module_state["model"],
        #                     optimizer_key=optim_key,
        #                     storage_reader=reader,
        #                 )
        #                 flattened_osd = FSDP.optim_state_dict_to_load(
        #                     optim_state_dict=optim_state[optim_key],
        #                     model=self.model,
        #                     optim=optim,
        #                 )
        #                 optim.load_state_dict(flattened_osd)
        #
        #     # Load metadata (anything not a module or optimizer)
        #     metadata = torch.load(path / _METADATA_FILENAME)
        #     return metadata
        #
        # if _is_full_checkpoint(path):
        #     checkpoint = _lazy_load(path)
        #     _load_raw_module_state(
        #         checkpoint.pop("state_dict"),
        #         module=self.model,
        #         world_size=self.world_size,
        #         strict=self.lightning_module.strict_loading,
        #     )
        #
        #     # Materialize lazy tensors if there are any left in the checkpoint
        #     # The `torch.Optimizer.load_state_dict` method can't load lazy tensors because of deepcopy pickle issues
        #     checkpoint = _materialize_tensors(checkpoint)
        #
        #     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        #     from torch.distributed.fsdp import OptimStateKeyType
        #
        #     optimizer_states = checkpoint.get("optimizer_states")
        #     if optimizer_states is None or self.lightning_module.trainer.state.fn != TrainerFn.FITTING:
        #         # If the optimizer states are not present, we don't need to do anything (backward compatibility)
        #         return checkpoint
        #     if len(self.optimizers) != len(optimizer_states):
        #         raise RuntimeError(
        #             f"You have configured {len(self.optimizers)} optimizers but the checkpoint contains"
        #             f" {len(optimizer_states)} optimizers to load. Please resume training with the same number"
        #             " of optimizers or edit the checkpoint manually to remove states."
        #         )
        #
        #     # rank0_only should be false because we need to load the optimizer state on all ranks
        #     with _get_full_state_dict_context(self.model, world_size=self.world_size, rank0_only=False):
        #         for optimizer, opt_state in zip(self.optimizers, optimizer_states):
        #             if isinstance(list(opt_state["state"].keys())[0], int):
        #                 # Handling the case where the optimizer state is saved from a normal optimizer
        #                 opt_state = FSDP.rekey_optim_state_dict(opt_state, OptimStateKeyType.PARAM_NAME, self.model)
        #
        #             opt_state = FSDP.optim_state_dict_to_load(
        #                 optim_state_dict=opt_state,
        #                 model=self.model,
        #                 optim=optimizer,
        #             )
        #             optimizer.load_state_dict(opt_state)
        #
        #     return checkpoint
        #
        # raise ValueError(
        #     f"The path {str(path)!r} does not point to a valid checkpoint. Make sure the path points to either a"
        #     " directory with FSDP checkpoint shards, or a single file with a full checkpoint."
        # )

    def _setup_distributed(self) -> None:
        super().setup_environment()
        reset_seed()
        self.set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank

