import os 
import torch
import torch.nn as nn 
import random
import json 
import math
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from .ps_utils import find_match_prob, sort_voiced_unvoiced, find_keys, find_ps_keys

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
        self.upstream = upstream
        self.prune_config = prune_config

        self.num_layers = len(self.upstream.model.encoder.layers)
        each_step = self.prune_config["prune"]["num_rows_each_step"]
        self.num_rows_each_step = [each_step for i in range(self.num_layers)] if type(each_step)==int else each_step

        # Assume that the number of dimension 
        ffn_dim = self.upstream.model.encoder.ffn_embedding_dim
        self.total_ffn_dim = [ffn_dim for i in range(self.num_layers)] if type(ffn_dim)==int else ffn_dim
        self.upstream.model.encoder.ffn_embedding_dim = self.total_ffn_dim[:]
        self.total_prune_step = self.prune_config["prune"]["total_steps"]
        
        self.prune_type = self.prune_config["prune"]["prune_type"]
        self.prune_strategy = self.prune_config["prune"]["prune_strategy"]
        self.prune_min = self.prune_config["prune"]["prune_min"]

        self.data_info = self.prune_config["prune"]["data_info"]
        

    def prune_api(self):
        prune_dim = self.prune(self.upstream.model.encoder)
        for i in range(self.num_layers):
            self.total_ffn_dim[i] -= prune_dim[i]
            tqdm.write(f"[Row Pruning] {self.total_ffn_dim[i]} hidden dimension are remained {i+1} layer's fead forward network")
        # if 'melhubert' in self.upstream.upstream_config:
        #     self.upstream.upstream_config['melhubert']['encoder_ffn_embed_dim'] = self.total_ffn_dim
        # else:
        #     self.upstream.upstream_config['hubert']['encoder_ffn_embed_dim'] = self.total_ffn_dim
        return self.total_ffn_dim

    def prune(self, encoder):
        n_to_prune = self.num_rows_each_step

        if self.prune_type == 'protect' and self.prune_strategy == 'all':
            protect_rows = self.find_protect_keys(self.data_info, layer=-1)

        prune_dim = []
        for layer in range(self.num_layers):
            rows_and_score = self.get_layer_rows_norm(encoder.layers[layer].fc1, encoder.layers[layer].fc2, layer)
            if self.prune_type == 'protect':
                if self.prune_strategy == 'all':
                    pr = protect_rows[layer]
                if self.prune_strategy == 'each':
                    pr = self.find_protect_keys(self.data_info, layer=layer)
                rows_and_score = [r for r in rows_and_score if r[0] not in pr]
            rows_and_score = sorted(rows_and_score, key=lambda x:x[1])
            sorted_rows = [row_and_score[0] for row_and_score in rows_and_score]
            if n_to_prune[layer] != -1:
                to_prune = sorted_rows[:n_to_prune[layer]]
            else:
                to_prune = sorted_rows[:]
            if len(to_prune):
                if self.total_ffn_dim[layer] == len(to_prune):
                    to_prune = random.sample(to_prune, len(to_prune)-self.prune_min)
                self.prune_layer_ffn(encoder.layers[layer].fc1, encoder.layers[layer].fc2, to_prune, layer)
            prune_dim.append(len(to_prune))
            self.upstream.model.encoder.ffn_embedding_dim[layer] -= len(to_prune)

        return prune_dim

    def find_protect_keys(self, data_info, layer=-1):
        phone_label_pth = data_info['phone_label_pth']
        with open(phone_label_pth, 'r') as fp:
            phone_label = json.load(fp)
        sort_phone_unvoiced, num_type = sort_voiced_unvoiced(phone_label)
        split_idx = [0]+[sum(num_type[:i+1]) for i in range(len(num_type))]
        phone_idx = [phone_label[x][0] for x in sort_phone_unvoiced]
        match_prob = find_match_prob(self.upstream.model, data_info, layer)
        keys = find_keys(match_prob, phone_idx, split_idx, layer)
        ps_keys = find_ps_keys(keys, layer)
        properties = ps_keys.keys()

        protect_keys = []
        for idx in range(len(self.upstream.model.encoder.layers)):
            pk = set([])
            for p in properties:
                pk = pk | set(ps_keys[p][idx+1])
            protect_keys.append(list(pk))
            if layer == -1 or layer == idx:
                print(f"There are {len(pk)} total ps keys in layer {idx+1}")
                for p in properties:
                    print(f"There are {len(ps_keys[p][idx+1])} total {p} ps keys in layer {idx+1}")
                    
        return protect_keys if layer == -1 else protect_keys[layer]

    def prune_layer_ffn(self, fc1, fc2, to_prune, layer):
        new_fc1_weight = []
        new_fc1_bias = []
        new_fc2_weight = []

        for i in range(self.total_ffn_dim[layer]):
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
       
        new_hidden_dim = self.total_ffn_dim[layer] - len(to_prune)
        
        fc1.out_features = new_hidden_dim
        fc2.in_features = new_hidden_dim
        return 
    
    def get_layer_rows_norm(self, fc1, fc2, layer):
        assert isinstance(fc1, nn.Linear)
        assert isinstance(fc2, nn.Linear)
        
        fc1_norm = []
        fc2_norm = []

        for i in range(self.total_ffn_dim[layer]):
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
        for i in range(self.total_ffn_dim[layer]):
            norm = fc1_norm[i] + fc2_norm[i]
            ffn_norm.append((i, norm))
        return ffn_norm
