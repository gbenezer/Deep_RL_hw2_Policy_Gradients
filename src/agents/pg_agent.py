from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray
import torch

from networks.critics import ValueCritic
from networks.policies import MLPPolicyPG
from infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline and baseline_learning_rate is not None:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[NDArray],
        actions: Sequence[NDArray],
        rewards: Sequence[NDArray[np.floating]],
        terminals: Sequence[NDArray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[NDArray[np.floating]] = self._calculate_q_vals(rewards)

        # flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.

        flat_obs = np.concatenate(obs)
        flat_actions = np.concatenate(actions)
        flat_rewards = np.concatenate(rewards)
        flat_terminals = np.concatenate(terminals)
        flat_q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            flat_obs, flat_rewards, flat_q_values, flat_terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        info: dict = None

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info = None

            info.update(critic_info)

        return info

    def _discounted_return(self, rewards: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Helper function which takes a numpy array of floats of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a numpy array of floats where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """

        # get the time horizon
        horizon = len(rewards)

        # early exit for short trajectories
        if horizon < 2:
            return rewards

        # create the discounting exponents
        exponents = np.arange(horizon)

        # sum the discounted rewards from beginning to end
        discounted_reward_sum = np.sum(np.power(rewards, exponents))

        return np.repeat(discounted_reward_sum, repeats=horizon)

    def _discounted_reward_to_go(
        self, rewards: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Helper function which takes a numpy array of floats of rewards {r_0, r_1, ..., r_t', ... r_T}
        and returns a numpy array of floats where the entry in each index t is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """

        # if there is zero or one rewards,
        # return the rewards unchanged
        if len(rewards) < 2:
            return rewards

        reward_to_go = rewards.copy()

        # from the penultimate entry of the rewards list, iterating backwards
        # calculate reward to go by iteratively discounting reward to go
        for t in range(len(rewards) - 2, -1, -1):
            reward_to_go[t] = rewards[t + 1] + self.gamma * reward_to_go[t + 1]

        return reward_to_go

    def _calculate_q_vals(
        self, rewards: Sequence[NDArray[np.floating]]
    ) -> Sequence[NDArray[np.floating]]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}

            q_values = [
                self._discounted_return(reward_trajectory)
                for reward_trajectory in rewards
            ]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}

            q_values = [
                self._discounted_reward_to_go(reward_trajectory)
                for reward_trajectory in rewards
            ]

        return q_values

    def _estimate_advantage(
        self,
        obs: NDArray,
        rewards: NDArray[np.floating],
        q_values: NDArray,
        terminals: NDArray,
    ) -> NDArray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            advantages = q_values
            advantages_tensor = None
        else:

            # convert ndarrays to torch tensors on-device
            obs_tensor = torch.from_numpy(obs).float().to(device=ptu.device)
            rewards_tensor = torch.from_numpy(rewards).float().to(device=ptu.device)
            q_values_tensor = torch.from_numpy(q_values).float().to(device=ptu.device)
            terminals_tensor = torch.from_numpy(terminals).float().to(device=ptu.device)

            values_tensor = self.critic.forward(obs=obs_tensor)
            assert values_tensor.size() == q_values_tensor.size()

            if self.gae_lambda is None:

                advantages_tensor = q_values_tensor - values_tensor
                advantages = advantages_tensor.numpy(force=True)
            else:
                batch_size = obs_tensor.size()[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values_tensor = torch.concatenate((values_tensor, torch.zeros(1)))
                advantages_tensor = torch.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):

                    # if the state is terminal, the advantage is defined to be zero
                    if terminals_tensor[i] == 1:
                        continue

                    # otherwise calculate the sum recursively
                    else:
                        
                        # Temporal difference estimate of the advantage
                        advantage_td_estimate = (
                            rewards_tensor[i]
                            + self.gamma * values_tensor[i + 1]
                            - values_tensor[i]
                        )
                        
                        # Recursive application of lambda and gamma
                        advantages_tensor[i] = (
                            advantage_td_estimate
                            + self.gamma * self.gae_lambda * advantages_tensor[i + 1]
                        )

                # remove dummy advantage
                advantages_tensor = advantages_tensor[:-1]
                advantages = advantages_tensor.numpy(force=True)

        if self.normalize_advantages and advantages is not None:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-12)

        return advantages
