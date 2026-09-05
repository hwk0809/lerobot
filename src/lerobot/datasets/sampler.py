#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Iterator

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


def make_cotrain_sampler(
    dataset,
    real_w: float,
    split_episode: int,
) -> "torch.utils.data.WeightedRandomSampler":
    """Sim/real co-training sampler for a dataset merged by `aggregate_datasets`.

    Episodes `[0, split_episode)` are the simulation half and `[split_episode, N)` the real
    half (that is the order `tools/make_cotrain_dataset.py` writes, and it records
    `split_episode` in `meta/cotrain_meta.json`). Every drawn sample lands in the real half
    with probability `real_w`, uniformly within each half.

    This is the per-sample reweighting used in the sim-and-real co-training literature
    (arXiv:2503.24361), not a hard per-batch quota: a batch contains real samples only in
    expectation. With `real_w` left at the natural frame ratio the sampler is equivalent to
    plain shuffling, so it only earns its keep when sweeping the ratio.
    """
    from_indices = dataset.meta.episodes["dataset_from_index"]
    to_indices = dataset.meta.episodes["dataset_to_index"]
    num_episodes = len(from_indices)

    if not 0.0 < real_w < 1.0:
        raise ValueError(f"cotrain_w must be in (0, 1), got {real_w}")
    if not 0 < split_episode < num_episodes:
        raise ValueError(
            f"cotrain_split_episode must be in (0, {num_episodes}), got {split_episode}"
        )

    spans = [(int(f), int(t)) for f, t in zip(from_indices, to_indices, strict=True)]
    sim_frames = sum(t - f for (f, t) in spans[:split_episode])
    real_frames = sum(t - f for (f, t) in spans[split_episode:])
    if sim_frames == 0 or real_frames == 0:
        raise ValueError(f"empty half: sim_frames={sim_frames} real_frames={real_frames}")

    total_frames = sim_frames + real_frames
    weights = torch.zeros(total_frames, dtype=torch.double)
    for episode_idx, (start, end) in enumerate(spans):
        if episode_idx < split_episode:
            weights[start:end] = (1.0 - real_w) / sim_frames
        else:
            weights[start:end] = real_w / real_frames

    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=total_frames, replacement=True
    )
