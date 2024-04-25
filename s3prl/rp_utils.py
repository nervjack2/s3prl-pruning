import os 
import torch
import torch.nn as nn 
import math
from fairseq_code import MultiheadAttention
from tqdm import tqdm
from collections import defaultdict
from dataset import MelFeatDataset
from torch.utils.data import DataLoader

def set_prune_interval(prune_interval, warm_up_steps, total_prune_steps):
    if isinstance(prune_interval, int):
        tmp = [prune_interval*i for i in range(total_prune_steps)]
        prune_interval = [warm_up_steps+p for p in tmp]        
    elif isinstance(prune_interval, list):
        prune_interval = [warm_up_steps+p for p in prune_interval]
    else:
        raise NotImplementedError

    return prune_interval

class RowPruningTools():
    def __init__(self, device, prune_config, upstream):
        self.device = device
        self.prune_config = prune_config
        self.upstream = upstream

        self.num_layers = len(self.upstream.model.encoder.layers)
        self.num_rows_each_step = self.prune_config["num_rows_each_step"]

        self.total_ffn_dim = self.upstream.model.encoder.ffn_embedding_dim

        self.total_prune_step = self.prune_config["total_steps"]
        assert self.num_rows_each_step * self.total_prune_step <= self.total_ffn_dim

    def prune_api(self):
        self.prune(self.upstream.model.encoder)
        self.total_ffn_dim -= self.num_rows_each_step
        self.upstream.upstream_config['melhubert']['encoder_ffn_embed_dim'] = self.total_ffn_dim
        tqdm.write(f"[Row Pruning] {self.total_ffn_dim} hidden dimension are remained in fead forward network")

    def prune(self, encoder):
        n_to_prune = self.num_rows_each_step

        for layer in range(self.num_layers):
            rows_and_score = self.get_layer_rows_norm(encoder.layers[layer].fc1, encoder.layers[layer].fc2, layer)
            rows_and_score = sorted(rows_and_score, key=lambda x:x[1])
            sorted_rows = [row_and_score[0] for row_and_score in rows_and_score]
            to_prune = sorted_rows[:n_to_prune]
            self.prune_layer_ffn(encoder.layers[layer].fc1, encoder.layers[layer].fc2, to_prune)
    
    def prune_layer_ffn(self, fc1, fc2, to_prune):
        new_fc1_weight = []
        new_fc1_bias = []
        new_fc2_weight = []

        for i in range(self.total_ffn_dim):
            if i not in to_prune:
                new_fc1_weight.append(
                        fc1.weight[i,:].unsqueeze(0)
                    )
                new_fc1_bias.append(fc1.bias[i])

                new_fc2_weight.append(
                   fc2.weight[:,i].unsqueeze(1)
                )

        new_fc1_weight = torch.cat(new_fc1_weight).detach()
        new_fc1_bias = torch.Tensor(new_fc1_bias).to(self.device).detach()
        new_fc2_weight = torch.cat(new_fc2_weight, dim=1).detach()
    
        new_fc1_weight.requires_grad = True
        new_fc1_bias.requires_grad = True
        new_fc2_weight.requires_grad = True

        fc1.weight = torch.nn.Parameter(new_fc1_weight)
        fc1.bias = torch.nn.Parameter(new_fc1_bias)
        fc2.weight = torch.nn.Parameter(new_fc2_weight)
       
        new_hidden_dim = self.total_ffn_dim - self.num_rows_each_step 
        
        fc1.out_features = new_hidden_dim
        fc2.in_features = new_hidden_dim
        return 
    
    def get_layer_rows_norm(self, fc1, fc2, layer):
        assert isinstance(fc1, nn.Linear)
        assert isinstance(fc2, nn.Linear)
        
        fc1_norm = []
        fc2_norm = []

        for i in range(self.total_ffn_dim):
            fc1_norm.append(
                torch.sum(
                    torch.abs(
                        fc1.weight[i,:]
                    )
                ).tolist()
                + torch.abs(fc1.bias[i]).tolist()
            )
            fc2_norm.append(
                torch.sum(
                    torch.abs(
                        fc2.weight[:,i]
                    )
                ).tolist()
            )

        ffn_norm = []
        for i in range(self.total_ffn_dim):
            norm = fc1_norm[i] + fc2_norm[i]
            ffn_norm.append((i, norm))
        return ffn_norm
     