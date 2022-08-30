from typing import List

import torch
import torch.nn as nn

from s3prl import Container, Output

from . import NNModule


class FrameLevel(NNModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [256],
        activation: str = None,
        **unused,
    ):
        """
        activation: ReLU
        """
        super().__init__()
        hidden_sizes = hidden_sizes
        latest_size = input_size

        hidden_layers = []
        if len(hidden_sizes) > 0:
            for size in hidden_sizes:
                hidden_layers.append(nn.Linear(latest_size, size))
                if activation is not None:
                    hidden_layers.append(getattr(nn, activation)())
                latest_size = size

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.final_proj = nn.Linear(latest_size, output_size)

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x, x_len=None):
        ys = self.hidden_layers(x)
        ys = self.final_proj(ys)
        return ys, x_len


class UtteranceLevel(NNModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [256],
        activation: str = None,
        pooling_cfg: dict = dict(
            CLS="MeanPooling",
        ),
        **unused,
    ):
        super().__init__()
        pooling_cfg = pooling_cfg
        hidden_sizes = hidden_sizes
        latest_size = input_size

        hidden_layers = []
        if len(hidden_sizes) > 0:
            for size in hidden_sizes:
                hidden_layers.append(nn.Linear(latest_size, size))
                if activation is not None:
                    hidden_layers.append(getattr(nn, activation)())
                latest_size = size

        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.pooling = Container(pooling_cfg)()
        self.final_proj = nn.Linear(latest_size, output_size)

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x, x_len=None):
        x = self.hidden_layers(x)
        x_pooled = self.pooling(x, x_len)
        y = self.final_proj(x_pooled)
        return y
