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
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import ModelParallelStrategy
from torch.utils.data import DataLoader, DistributedSampler

from tests_fabric.helpers.datasets import RandomDataset
from tests_fabric.helpers.runif import RunIf


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
            strategy=strategy
        )

        device_mesh = trainer.strategy.device_mesh
        assert isinstance(device_mesh, DeviceMesh)
        assert device_mesh.device_type == model.device.type
        assert device_mesh.mesh_dim_names == ("data_parallel", "tensor_parallel")
        assert device_mesh.size(0) == dp_size
        assert device_mesh.size(1) == tp_size
        assert device_mesh.ndim == 2

        trainer.strategy.barrier()

    # Passing "auto" will select internode and intranode dimensions automatically
    strategy = ModelParallelStrategy(
        data_parallel_size="auto",
        tensor_parallel_size="auto",
    )
    fabric = Trainer(
        accelerator="auto",
        devices=4,
        num_nodes=1,
        strategy=strategy
    )
    assert fabric.strategy.device_mesh.mesh_dim_names == ("data_parallel", "tensor_parallel")
    assert fabric.strategy.device_mesh.size(0) == 1
    assert fabric.strategy.device_mesh.size(1) == 4


@RunIf(min_torch="2.3", standalone=True, min_cuda_gpus=2)
def test_tensor_parallel():
    from torch.distributed._tensor import DTensor

    strategy = ModelParallelStrategy(parallelize_fn=_parallelize_feed_forward_tp)
    fabric = Fabric(accelerator="auto", devices=2, strategy=strategy)
    fabric.launch()

    fabric.seed_everything(0)

    with fabric.init_module(empty_init=True):
        model = FeedForward()

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)

    device_mesh = fabric.strategy.device_mesh
    assert all(tensor.device_mesh == device_mesh["tensor_parallel"] for tensor in optimizer.param_groups[0]["params"])
    assert all(isinstance(weight, DTensor) for weight in model.parameters())
    assert model.w1.weight.device_mesh == device_mesh["tensor_parallel"]

    dataset_size = 6
    dataset = RandomDataset(32, dataset_size)
    dataloader = DataLoader(dataset, batch_size=2)
    dataloader = fabric.setup_dataloaders(dataloader)

    # No data sharding, all GPUs get the same input inside a TP group
    assert len(dataloader) == dataset_size // dataloader.batch_size
    assert isinstance(dataloader.sampler, DistributedSampler)

    for _, batch in enumerate(dataloader):
        # All batches must be identical across TP group
        batches = fabric.all_gather(batch)
        assert all(torch.equal(batches[0], batches[i]) for i in range(1, len(batches)))

        output = model(batch)
        fabric.backward(output.sum())
        assert isinstance(model.w1.weight.grad, DTensor)
        assert model.w1.weight.grad.device_mesh == device_mesh["tensor_parallel"]
        optimizer.step()
        optimizer.zero_grad()


@RunIf(min_torch="2.3", standalone=True, min_cuda_gpus=4)
def test_fsdp2_tensor_parallel():
    from torch.distributed._tensor import DTensor

    strategy = ModelParallelStrategy(
        parallelize_fn=_parallelize_feed_forward_fsdp2_tp,
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    fabric = Fabric(accelerator="auto", devices=4, strategy=strategy)
    fabric.launch()

    fabric.seed_everything(0)

    with fabric.init_module(empty_init=True):
        model = FeedForward()

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)

    assert all(isinstance(weight, DTensor) for weight in model.parameters())
    assert all(isinstance(tensor, DTensor) for tensor in optimizer.param_groups[0]["params"])
    assert model.w1.weight.device_mesh.ndim == 2
    assert model.w1.weight.device_mesh.size(0) == 2
    assert model.w1.weight.device_mesh.size(1) == 2
    assert all(weight.device.type != "meta" for weight in model.parameters())
    assert all(tensor.device_mesh.ndim == 2 for tensor in optimizer.param_groups[0]["params"])
    assert all(tensor.device.type != "meta" for tensor in optimizer.param_groups[0]["params"])

    dataset_size = 8
    dataset = RandomDataset(32, dataset_size)
    dataloader = DataLoader(dataset, batch_size=2)
    dataloader = fabric.setup_dataloaders(dataloader)

    # No data sharding across TP dimension, sharding across data-parallel dimension only
    device_mesh = fabric.strategy.device_mesh
    dp_mesh = device_mesh["data_parallel"]
    tp_mesh = device_mesh["tensor_parallel"]
    assert len(dataloader) == dataset_size // dataloader.batch_size // dp_mesh.size()
    assert isinstance(dataloader.sampler, DistributedSampler)

    for _, batch in enumerate(dataloader):
        batches = fabric.all_gather(batch)
        # Batches across the TP dimension must be identical
        batches_tp = batches[tp_mesh.mesh]
        assert all(torch.equal(batches_tp[0], batches_tp[i]) for i in range(1, len(batches_tp)))
        # Batches across the DP dimension must be different
        batches_dp = batches[dp_mesh.mesh]
        assert all(not torch.equal(batches_dp[0], batches_dp[i]) for i in range(1, len(batches_dp)))

        output = model(batch)
        fabric.backward(output.sum())
        assert isinstance(model.w1.weight.grad, DTensor)
        assert model.w1.weight.grad.device_mesh == device_mesh
        optimizer.step()
        optimizer.zero_grad()
