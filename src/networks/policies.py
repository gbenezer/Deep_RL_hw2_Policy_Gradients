import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
from numpy.typing import NDArray
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            # neural network with raw logits as output
            self.net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.net.parameters()
        else:
            
            # neural network with Gaussian/Normal action distribution means as output
            self.net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            
            # separate parameter for learnable log standard deviations
            # changed default to e to have initialization with unit standard dev
            self.logstd = nn.Parameter(
                torch.repeat_interleave(
                    input=torch.as_tensor(
                        data=np.e, 
                        dtype=torch.float32, 
                        device=ptu.device), 
                    repeats=ac_dim)
            )
            
            parameters = itertools.chain([self.logstd], self.net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: NDArray) -> NDArray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        action = None

        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            action_logits: torch.Tensor = self.net(obs)
            return D.Categorical(logits=action_logits)
        else:
            action_means: torch.Tensor = self.net(obs)
            action_stdev = torch.exp(self.logstd)
            return D.Normal(loc=action_means, scale=action_stdev)

    def update(self, obs: NDArray, actions: NDArray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: NDArray,
        actions: NDArray,
        advantages: NDArray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: compute the policy gradient actor loss
        loss = None

        # TODO: perform an optimizer step
        pass

        return {
            "Actor Loss": loss.item(),
        }
