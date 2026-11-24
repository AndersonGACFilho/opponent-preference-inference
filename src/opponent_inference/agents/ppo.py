"""PPO (clipped, GAE) for the omega-conditioned MO-Connect4 agent.

The agent is single-objective on the surface: its reward is the scalarization `omega . R` at
the end of the game, and `omega` (its own preference vector) is part of the observation, so a
single network serves every omega. Default hyperparameters follow Appendix A of the base paper
(Obiang & Radulescu, 2026): 64-unit MLP, lr 0.01 with decay, GAE 0.95, clip 0.2, entropy 0.01,
gamma 1, 4 environments, 256 steps per rollout, 4 minibatches, 4 epochs.

`train_ppo` takes the environments and the observation function (`obs_fn`) as arguments, so the
oracle and the omega-hat agent can reuse it unchanged with a different input vector.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..env import ConnectFourOpponentEnv, flatten_obs
from .policy import ObsFn, PolicyNet


@dataclass
class PPOConfig:
    total_steps: int = 2_000_000
    n_envs: int = 4
    n_steps: int = 256
    n_minibatches: int = 4
    n_epochs: int = 4
    lr: float = 0.01
    lr_decay: bool = True          # linear decay to 0 over training
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    seed: int = 0


LOG_FIELDS = [
    "update", "global_step", "mean_utility", "win_rate", "n_episodes",
    "policy_loss", "value_loss", "entropy", "lr", "seconds",
]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation. Arrays are (n_steps, n_envs); returns (adv, returns).

    `dones[t]` is 1 if the episode ended at step t (the next value is then not bootstrapped).
    """
    n_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros_like(last_value)
    for t in reversed(range(n_steps)):
        next_value = last_value if t == n_steps - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def train_ppo(
    envs: Sequence[ConnectFourOpponentEnv],
    cfg: PPOConfig,
    obs_fn: ObsFn = flatten_obs,
    log_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> PolicyNet:
    """Train a `PolicyNet` with PPO on `envs` (one per parallel actor); return it in eval mode.

    The environments must return the vector reward; the scalar reward `omega . R` is computed
    here from each environment's own omega. Episodes are reset manually when they end.
    """
    if len(envs) != cfg.n_envs:
        raise ValueError(f"expected {cfg.n_envs} environments, got {len(envs)}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    obs = [env.reset(seed=cfg.seed * 1000 + i)[0] for i, env in enumerate(envs)]
    input_dim = obs_fn(obs[0]).shape[-1]
    n_actions = envs[0].action_space.n
    net = PolicyNet(input_dim, n_actions, cfg.hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    batch_size = cfg.n_envs * cfg.n_steps
    minibatch_size = batch_size // cfg.n_minibatches
    n_updates = cfg.total_steps // batch_size

    buf_x = np.zeros((cfg.n_steps, cfg.n_envs, input_dim), dtype=np.float32)
    buf_mask = np.zeros((cfg.n_steps, cfg.n_envs, n_actions), dtype=np.int8)
    buf_action = np.zeros((cfg.n_steps, cfg.n_envs), dtype=np.int64)
    buf_logp = np.zeros((cfg.n_steps, cfg.n_envs), dtype=np.float32)
    buf_value = np.zeros((cfg.n_steps, cfg.n_envs), dtype=np.float32)
    buf_reward = np.zeros((cfg.n_steps, cfg.n_envs), dtype=np.float32)
    buf_done = np.zeros((cfg.n_steps, cfg.n_envs), dtype=np.float32)

    recent_utility: List[float] = []
    recent_win: List[float] = []
    writer = None
    log_file = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", newline="")
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        writer.writeheader()

    start = time.perf_counter()
    global_step = 0
    try:
        for update in range(n_updates):
            lr = cfg.lr * (1.0 - update / n_updates) if cfg.lr_decay else cfg.lr
            for group in optimizer.param_groups:
                group["lr"] = lr

            # ---- rollout
            for t in range(cfg.n_steps):
                x = np.stack([obs_fn(o) for o in obs]).astype(np.float32)
                mask = np.stack([o["action_mask"] for o in obs])
                with torch.no_grad():
                    action, logp, _, value = net.act(torch.from_numpy(x), torch.from_numpy(mask))
                buf_x[t], buf_mask[t] = x, mask
                buf_action[t] = action.numpy()
                buf_logp[t] = logp.numpy()
                buf_value[t] = value.numpy()

                for i, env in enumerate(envs):
                    omega = env.omega
                    next_obs, reward, terminated, truncated, _ = env.step(int(action[i]))
                    done = terminated or truncated
                    buf_done[t, i] = float(done)
                    if done:
                        scalar = float(np.dot(omega, reward))
                        buf_reward[t, i] = scalar
                        recent_utility.append(scalar)
                        recent_win.append(float(reward[0] > 0))
                        next_obs, _ = env.reset()
                    else:
                        buf_reward[t, i] = 0.0
                    obs[i] = next_obs
                global_step += cfg.n_envs

            # ---- advantages
            with torch.no_grad():
                x_last = torch.from_numpy(np.stack([obs_fn(o) for o in obs]).astype(np.float32))
                last_value = net.value(x_last).numpy()
            advantages, returns = compute_gae(
                buf_reward, buf_value, buf_done, last_value, cfg.gamma, cfg.gae_lambda
            )

            # ---- optimization
            b_x = torch.from_numpy(buf_x.reshape(batch_size, input_dim))
            b_mask = torch.from_numpy(buf_mask.reshape(batch_size, n_actions))
            b_action = torch.from_numpy(buf_action.reshape(batch_size))
            b_logp = torch.from_numpy(buf_logp.reshape(batch_size))
            b_adv = torch.from_numpy(advantages.reshape(batch_size))
            b_ret = torch.from_numpy(returns.reshape(batch_size))
            pg_losses, v_losses, entropies = [], [], []
            for _ in range(cfg.n_epochs):
                permutation = torch.randperm(batch_size)
                for start_idx in range(0, batch_size, minibatch_size):
                    idx = permutation[start_idx:start_idx + minibatch_size]
                    _, new_logp, entropy, new_value = net.act(b_x[idx], b_mask[idx], b_action[idx])
                    ratio = (new_logp - b_logp[idx]).exp()
                    adv = b_adv[idx]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    pg_loss = torch.max(
                        -adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
                    ).mean()
                    v_loss = 0.5 * ((new_value - b_ret[idx]) ** 2).mean()
                    loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    entropies.append(entropy.mean().item())

            # ---- logging
            window_u, window_w = recent_utility[-200:], recent_win[-200:]
            row = {
                "update": update + 1,
                "global_step": global_step,
                "mean_utility": float(np.mean(window_u)) if window_u else float("nan"),
                "win_rate": float(np.mean(window_w)) if window_w else float("nan"),
                "n_episodes": len(recent_utility),
                "policy_loss": float(np.mean(pg_losses)),
                "value_loss": float(np.mean(v_losses)),
                "entropy": float(np.mean(entropies)),
                "lr": lr,
                "seconds": time.perf_counter() - start,
            }
            if writer is not None:
                writer.writerow(row)
                log_file.flush()
            if verbose and ((update + 1) % 20 == 0 or update == 0 or update + 1 == n_updates):
                print(
                    f"[{update + 1}/{n_updates}] step {global_step} "
                    f"utility {row['mean_utility']:+.3f} win {row['win_rate']:.2f} "
                    f"entropy {row['entropy']:.3f} ({row['seconds']:.0f}s)",
                    flush=True,
                )
    finally:
        if log_file is not None:
            log_file.close()

    net.eval()
    return net


def config_dict(cfg: PPOConfig) -> Dict[str, object]:
    """Config as a plain dict (stored next to checkpoints)."""
    return asdict(cfg)
