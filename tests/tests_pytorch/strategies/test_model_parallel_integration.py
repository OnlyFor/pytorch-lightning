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
import os
from copy import deepcopy
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Trainer, LightningModule, seed_everything
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.strategies import ModelParallelStrategy
from torch.utils.data import DataLoader, DistributedSampler

from tests_pytorch.helpers.runif import RunIf


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(32, 64)
        self.w2 = nn.Linear(32, 64)
        self.w3 = nn.Linear(64, 32)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def _parallelize_feed_forward_tp(model, device_mesh):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    tp_mesh = device_mesh["tensor_parallel"]
    tp_plan = {
        "w1": ColwiseParallel(),
        "w2": ColwiseParallel(),
        "w3": RowwiseParallel(),
    }
    parallelize_module(model, tp_mesh, tp_plan)
    return model


def _parallelize_feed_forward_fsdp2(model, device_mesh):
    from torch.distributed._composable.fsdp.fully_shard import fully_shard

    dp_mesh = device_mesh["data_parallel"]
    assert dp_mesh.ndim == 1  # Hybrid-sharding not supported

    # Fully-shard each layer
    fully_shard(model.w1, mesh=dp_mesh)
    fully_shard(model.w2, mesh=dp_mesh)
    fully_shard(model.w3, mesh=dp_mesh)

    # TODO: Re-enable activation checkpointing
    # Currently, state dict keys get prefixed with '_checkpoint_wrapper' in the keys
    # which leads to mismatches when loading weights into a checkpoint-wrapped module.
    # PyTorch should handle this automatically.

    # model = checkpoint_wrapper(model)

    return model


def _parallelize_feed_forward_fsdp2_tp(model, device_mesh):
    model = _parallelize_feed_forward_tp(model, device_mesh)
    model = _parallelize_feed_forward_fsdp2(model, device_mesh)
    return model


@RunIf(min_torch="2.3", standalone=True, min_cuda_gpus=4)
def test_setup_device_mesh():
    from torch.distributed.device_mesh import DeviceMesh

    for dp_size, tp_size in ((1, 4), (4, 1), (2, 2)):
        strategy = ModelParallelStrategy(
            data_parallel_size=dp_size,
            tensor_parallel_size=tp_size,
        )
        trainer = Trainer(
            accelerator="auto",
            devices=4,
            strategy=strategy,
            logger=False,
            enable_checkpointing=False,
            max_steps=1,
        )

        class Model(BoringModel):
            def configure_model(self):
                device_mesh = self.device_mesh
                assert isinstance(device_mesh, DeviceMesh)
                assert device_mesh.device_type == model.device.type
                assert device_mesh.mesh_dim_names == ("data_parallel", "tensor_parallel")
                assert device_mesh.size(0) == dp_size
                assert device_mesh.size(1) == tp_size
                assert device_mesh.ndim == 2
        
        model = Model()
        trainer.fit(model)

    # Passing "auto" will select internode and intranode dimensions automatically
    strategy = ModelParallelStrategy(
        data_parallel_size="auto",
        tensor_parallel_size="auto",
    )
    trainer = Trainer(
        accelerator="auto",
        devices=4,
        num_nodes=1,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        max_steps=1,
    )

    class Model(BoringModel):
        def configure_model(self):
            device_mesh = self.device_mesh
            assert device_mesh.mesh_dim_names == ("data_parallel", "tensor_parallel")
            assert device_mesh.size(0) == 1
            assert device_mesh.size(1) == 4
        
    model = Model()
    trainer.fit(model)


@RunIf(min_torch="2.3", standalone=True, min_cuda_gpus=2)
def test_tensor_parallel():
    from torch.distributed._tensor import DTensor
    
    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.model = FeedForward()
        
        def configure_model(self):
            _parallelize_feed_forward_tp(self.model, device_mesh=self.device_mesh)
            
        def on_train_start(self):
            device_mesh = self.device_mesh
            optimizer = self.optimizers()
            assert all(tensor.device_mesh == device_mesh["tensor_parallel"] for tensor in optimizer.param_groups[0]["params"])
            assert all(isinstance(weight, DTensor) for weight in self.model.parameters())
            assert self.model.w1.weight.device_mesh == device_mesh["tensor_parallel"]
            
            # No data sharding, all GPUs get the same input inside a TP group
            dataloader = self.trainer.train_dataloader
            assert len(dataloader) == 6 // dataloader.batch_size
            assert isinstance(dataloader.sampler, DistributedSampler)
            
        def training_step(self, batch):
            # All batches must be identical across TP group
            batches = self.all_gather(batch)
            assert all(torch.equal(batches[0], batches[i]) for i in range(1, len(batches)))

            output = self.model(batch)
            return output.sum()
    
        def train_dataloader(self):
            dataset_size = 6
            dataset = RandomDataset(32, dataset_size)
            return DataLoader(dataset, batch_size=2)
            
        def configure_optimizers(self):
            return torch.optim.AdamW(model.parameters())

    trainer = Trainer(
        accelerator="auto", 
        devices=2, 
        strategy=ModelParallelStrategy(),
        max_steps=2,
        enable_checkpointing=False,
        logger=False,
    )

    seed_everything(0)
    with trainer.init_module(empty_init=True):
        model = Model()
        
    trainer.fit(model)


@RunIf(min_torch="2.3", standalone=True, min_cuda_gpus=4)
def test_fsdp2_tensor_parallel():
    from torch.distributed._tensor import DTensor

    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.model = FeedForward()
        
        def configure_model(self):
            _parallelize_feed_forward_fsdp2_tp(self.model, device_mesh=self.device_mesh)
            
        def on_train_start(self):
            optimizer = self.optimizers()
            assert all(isinstance(weight, DTensor) for weight in self.model.parameters())
            assert all(isinstance(tensor, DTensor) for tensor in optimizer.param_groups[0]["params"])
            assert self.model.w1.weight.device_mesh.ndim == 2
            assert self.model.w1.weight.device_mesh.size(0) == 2
            assert self.model.w1.weight.device_mesh.size(1) == 2
            assert all(weight.device.type != "meta" for weight in self.model.parameters())
            assert all(tensor.device_mesh.ndim == 2 for tensor in optimizer.param_groups[0]["params"])
            assert all(tensor.device.type != "meta" for tensor in optimizer.param_groups[0]["params"])
            
            # No data sharding across TP dimension, sharding across data-parallel dimension only
            device_mesh = self.device_mesh
            dp_mesh = device_mesh["data_parallel"]
            dataloader = self.trainer.train_dataloader
            assert len(dataloader) == 8 // dataloader.batch_size // dp_mesh.size()
            assert isinstance(dataloader.sampler, DistributedSampler)
            
        def training_step(self, batch):
            batches = self.all_gather(batch)
            dp_mesh = self.device_mesh["data_parallel"]
            tp_mesh = self.device_mesh["tensor_parallel"]
            
            # Batches across the TP dimension must be identical
            batches_tp = batches[tp_mesh.mesh]
            assert all(torch.equal(batches_tp[0], batches_tp[i]) for i in range(1, len(batches_tp)))
            # Batches across the DP dimension must be different
            batches_dp = batches[dp_mesh.mesh]
            assert all(not torch.equal(batches_dp[0], batches_dp[i]) for i in range(1, len(batches_dp)))

            output = self.model(batch)
            return output.sum()
    
        def train_dataloader(self):
            dataset_size = 8
            dataset = RandomDataset(32, dataset_size)
            return DataLoader(dataset, batch_size=2)
            
        def configure_optimizers(self):
            return torch.optim.AdamW(model.parameters())

    strategy = ModelParallelStrategy(
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    trainer = Trainer(
        accelerator="auto", 
        devices=4, 
        strategy=strategy,
        max_steps=2,
        enable_checkpointing=False,
        logger=False,
    )

    seed_everything(0)
    with trainer.init_module(empty_init=True):
        model = Model()
        
    trainer.fit(model)
