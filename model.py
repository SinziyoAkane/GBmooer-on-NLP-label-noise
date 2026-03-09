# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModel


class Bert4Classify(nn.Module):
    def __init__(self, pretrained_model_name_or_path, dropout_rate, num_classes, contrast=True):
        super(Bert4Classify, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.contrast = contrast
        d_model = 768 if 'bert' in pretrained_model_name_or_path else 1024
        self.projection_matrix = nn.Parameter(torch.randn(d_model, d_model)) if self.contrast else None

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, input_ids, att_mask):
        sentence_emb = self.get_sentence_embedding(input_ids, att_mask)
        output = self.classify(sentence_emb)
        return output

    def get_sentence_embedding(self, input_ids, att_mask):
        max_len = att_mask.sum(1).max()
        input_ids = input_ids[:, :max_len]
        att_mask = att_mask[:, :max_len]
        all_hidden = self.encoder(input_ids, att_mask)
        sentence_emb = all_hidden[0][:, 0]
        if self.contrast:
            return self.learnable_sphere_projection(sentence_emb)
        else:
            return sentence_emb

    def classify(self, x):
        output = self.mlp(x)
        return output

    def learnable_sphere_projection(self, input_ids):
        projected = torch.matmul(input_ids, self.projection_matrix)
        return projected / torch.norm(projected, p=2, dim=1, keepdim=True)

    def save_model(self, model_save_path):
        torch.save(self.state_dict(), model_save_path)

    def load_model(self, model_load_path):
        checkpoint = torch.load(model_load_path)
        self.encoder.load_state_dict(checkpoint, strict=False)

        if self.projection_matrix is not None:
            projection_key = 'projection_matrix'
            if projection_key in checkpoint:
                with torch.no_grad():
                    self.projection_matrix.copy_(checkpoint[projection_key])
            else:
                print(f"Warning: '{projection_key}' not found in checkpoint.")


