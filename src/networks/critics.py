import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.types import Number
from typing import Dict

import numpy as np
from numpy.typing import NDArray
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)

    def update(self, obs: NDArray, q_values: NDArray) -> Dict[str, Number]:
        
        obs_tensor = torch.from_numpy(obs).float().to(ptu.device)
        q_values_tensor = torch.from_numpy(q_values).float().to(ptu.device)
        
        critic_value_tensor = self.network.forward(input=obs_tensor)
        
        loss = F.smooth_l1_loss(input=critic_value_tensor, target=q_values_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Baseline Loss": loss.item(),
        }