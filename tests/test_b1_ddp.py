from __future__ import absolute_import, print_function

import math

import torch


def test_b1_ddp_batch_size_is_per_rank_and_sampler_is_partitioned():
    dataset_size = 19318
    world_size = 3
    per_rank_batch = 4
    dataset = list(range(dataset_size))
    samplers = [
        torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        for rank in range(world_size)
    ]
    indices = [list(iter(sampler)) for sampler in samplers]

    assert all(len(items) == int(math.ceil(
        float(dataset_size) / world_size
    )) for items in indices)
    assert all(
        len(set(indices[left][:per_rank_batch])
            & set(indices[right][:per_rank_batch])) == 0
        for left in range(world_size)
        for right in range(left + 1, world_size)
    )
    assert int(math.ceil(
        float(len(indices[0])) / per_rank_batch
    )) == 1610
    assert per_rank_batch * world_size == 12
