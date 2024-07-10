import numpy as np
import torch
from typing import Iterator
import logging

from torch.utils.data.sampler import BatchSampler


class EpisodicBatchSampler:
    def __init__(
        self, dataset, batch_sampler=None, n_way=5, k_shot=5, q_sample=15, episodes=600
    ):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_sample = q_sample
        self.episodes = episodes
        self.batch_size = n_way * (k_shot + q_sample)

        self.action_idx_list = dataset._action_idx
        self.action_idx_list = torch.LongTensor(self.action_idx_list)
        self.action_idx_set = list(set(self.action_idx_list.numpy()))
        self.action_idx_to_indices = {
            action_idx: np.where(self.action_idx_list.numpy() == action_idx)[0]
            for action_idx in self.action_idx_set
        }
        for l in self.action_idx_set:
            np.random.shuffle(self.action_idx_to_indices[l])
        self.used_action_idx_indices_count = {
            action_idx: 0 for action_idx in self.action_idx_set
        }

    def create_n_action_class_ids(self):
        action_ids = []
        while len(action_ids) < 5:
            id = np.random.randint(len(self.action_idx_set))
            if id not in action_ids:
                action_ids.append(id)
        return action_ids

    def __iter__(self) -> Iterator[int]:
        iteration = 0
        while iteration < self.episodes:
            action_ids = self.create_n_action_class_ids()
            indices = []

            for action_id in action_ids:
                start = self.used_action_idx_indices_count[action_id]
                end = self.used_action_idx_indices_count[action_id] + (
                    self.k_shot + self.q_sample
                )
                indices.extend(self.action_idx_to_indices[action_id][start:end])

                self.used_action_idx_indices_count[action_id] += (
                    self.k_shot + self.q_sample
                )
                # shuffle action_idx_indices_count list and reset count to zero
                if self.used_action_idx_indices_count[action_id] + (
                    self.k_shot + self.q_sample
                ) > len(self.action_idx_to_indices[action_id]):
                    np.random.shuffle(self.action_idx_to_indices[action_id])
                    self.used_action_idx_indices_count[action_id] = 0
            yield indices

    def __len__(self):
        return self.episodes
