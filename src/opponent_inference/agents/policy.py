"""Actor-critic MLP with action masking, shared by all learning agents.

The baseline MO-PPO, the oracle and the omega-hat agent use this same architecture; they only
differ in the input vector (`obs_fn`): board + own omega for the baseline, plus the true opponent
omega for the oracle, plus the inferred omega-hat for the omega-hat agent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..env import flatten_obs

ObsFn = Callable[[Dict[str, np.ndarray]], np.ndarray]
MASK_VALUE = -1e9


def _init_linear(layer: nn.Linear, std: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class PolicyNet(nn.Module):
    """Separate actor and critic MLPs (two hidden layers of `hidden` tanh units)."""

    def __init__(self, input_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        # Plain ints (not NumPy scalars) so checkpoints load with `weights_only=True`.
        self.input_dim, self.n_actions, self.hidden = int(input_dim), int(n_actions), int(hidden)
        root2 = float(np.sqrt(2.0))
        self.actor = nn.Sequential(
            _init_linear(nn.Linear(self.input_dim, self.hidden), root2), nn.Tanh(),
            _init_linear(nn.Linear(self.hidden, self.hidden), root2), nn.Tanh(),
            _init_linear(nn.Linear(self.hidden, self.n_actions), 0.01),
        )
        self.critic = nn.Sequential(
            _init_linear(nn.Linear(self.input_dim, self.hidden), root2), nn.Tanh(),
            _init_linear(nn.Linear(self.hidden, self.hidden), root2), nn.Tanh(),
            _init_linear(nn.Linear(self.hidden, 1), 1.0),
        )

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def distribution(self, x: torch.Tensor, mask: torch.Tensor) -> Categorical:
        """Action distribution; illegal columns (mask == 0) get probability zero."""
        logits = self.actor(x).masked_fill(mask == 0, MASK_VALUE)
        return Categorical(logits=logits)

    def act(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log-prob, entropy, value). Samples unless `action` is given."""
        dist = self.distribution(x, mask)
        if action is None:
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(x)


class PolicyAgent:
    """Callable `obs dict -> column` wrapper around a `PolicyNet`, for evaluation."""

    def __init__(self, net: PolicyNet, obs_fn: ObsFn = flatten_obs, deterministic: bool = True):
        self.net, self.obs_fn, self.deterministic = net, obs_fn, deterministic

    @torch.no_grad()
    def __call__(self, obs: Dict[str, np.ndarray]) -> int:
        x = torch.as_tensor(self.obs_fn(obs), dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(obs["action_mask"]).unsqueeze(0)
        action, *_ = self.net.act(x, mask, deterministic=self.deterministic)
        return int(action.item())


def save_policy(net: PolicyNet, path: str | Path, extra: dict | None = None) -> None:
    """Save weights plus the constructor arguments needed to rebuild the network."""
    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_dim": net.input_dim,
            "n_actions": net.n_actions,
            "hidden": net.hidden,
            "extra": extra or {},
        },
        path,
    )


def load_policy(path: str | Path) -> PolicyNet:
    """Rebuild a `PolicyNet` saved with `save_policy`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = PolicyNet(ckpt["input_dim"], ckpt["n_actions"], ckpt["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net
