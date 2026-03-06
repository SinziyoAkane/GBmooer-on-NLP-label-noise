# -*- coding: utf-8 -*-
import random

import torch.nn as nn
import torch
import torch.nn.functional as F


class UnsupervisedContrastiveLoss(nn.Module):
    def __init__(self, num_negatives, temperature=0.5):
        super(UnsupervisedContrastiveLoss, self).__init__()
        self.num_negatives = num_negatives
        self.temperature = temperature

    def forward(self, anchor, positive):
        # anchor: [batch_size, dim]
        # positive: [batch_size, dim]
        batch_size, dim = anchor.shape
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)  # [batch_size]

        neg_sim_list = []
        for i in range(batch_size):
            all_indices = list(range(batch_size))
            all_indices.remove(i)
            selected = random.sample(all_indices, min(self.num_negatives, batch_size - 1))
            neg_vectors = anchor[selected]  # [num_negatives, dim]

            anchor_i = anchor[i].unsqueeze(0).expand(len(selected), -1)  # [num_negatives, dim]
            sim = F.cosine_similarity(anchor_i, neg_vectors, dim=-1)  # [num_negatives]
            neg_sim_list.append(sim)

        neg_sim = torch.stack(neg_sim_list)  # [batch_size, num_negatives]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1 + num_negatives]
        logits = logits / self.temperature

        labels = torch.zeros(batch_size, dtype=torch.long).to(anchor.device)
        loss = F.cross_entropy(logits, labels)

        return loss
