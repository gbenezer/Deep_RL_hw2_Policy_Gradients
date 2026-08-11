import itertools
from torch import nn
import torch.distributions as D
from torch import optim
from typing import Dict
from torch.types import Number

import numpy as np
from numpy.typing import NDArray
import torch

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
            self.logstd = nn.Parameter(
                torch.zeros(size=(ac_dim,), dtype=torch.float32, device=ptu.device)
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

        # convert observation to a torch Float32 tensor on device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=ptu.device)

        # get the Distribution output of the network
        action_distribution: D.Distribution = self.forward(obs=obs_tensor)

        # sample a tensor from the output action distribution
        action_tensor = action_distribution.sample()

        return action_tensor.numpy(force=True)

    def forward(self, obs: torch.Tensor) -> D.Distribution:
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
    ) -> Dict[str, Number]:
        """Implements the policy gradient actor update."""
        obs_tensor = torch.from_numpy(obs).float().to(ptu.device)
        actions_tensor = torch.from_numpy(actions).to(ptu.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(ptu.device)

        # first get the actor action policy distribution
        observation_action_distribution = self.forward(obs=obs_tensor)

        # then get the negative log probability mass / density of the executed actions
        # under the observation action distribution (all positive numbers)
        negative_log_prob_actions = -1.0 * observation_action_distribution.log_prob(
            value=actions_tensor
        )
        
        # if the action space is not discrete, sum the pdf
        # across all dimensions of the action space
        if not self.discrete:
            negative_log_prob_actions = negative_log_prob_actions.sum(dim=-1)

        # the loss is the mean of the negative log probabilities weighted by the advantages
        loss = torch.mean(advantages_tensor * negative_log_prob_actions)

        # perform an optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }
