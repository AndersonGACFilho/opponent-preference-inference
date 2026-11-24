from pathlib import Path

import numpy as np
import pytest
import torch

from opponent_inference.agents.policy import PolicyAgent, PolicyNet, load_policy, save_policy
from opponent_inference.agents.ppo import PPOConfig, compute_gae, train_ppo
from opponent_inference.env import ConnectFourOpponentEnv, flat_obs_dim
from opponent_inference.solver import load_q_tables

TABLES = Path(__file__).resolve().parents[1] / "data" / "q_tables_4x4.npz"


@pytest.fixture(scope="module")
def tables():
    return load_q_tables(TABLES)


def test_gae_without_discount_spreads_terminal_reward():
    rewards = np.array([[0.0], [0.0], [1.0]])
    values = np.zeros((3, 1))
    dones = np.array([[0.0], [0.0], [1.0]])
    adv, ret = compute_gae(rewards, values, dones, np.zeros(1), gamma=1.0, gae_lambda=1.0)
    assert adv[:, 0] == pytest.approx([1.0, 1.0, 1.0])
    assert ret[:, 0] == pytest.approx([1.0, 1.0, 1.0])


def test_gae_does_not_bootstrap_across_episode_end():
    rewards = np.array([[1.0], [0.0]])
    values = np.zeros((2, 1))
    dones = np.array([[1.0], [0.0]])
    adv, _ = compute_gae(rewards, values, dones, np.array([5.0]), gamma=1.0, gae_lambda=0.95)
    assert adv[0, 0] == pytest.approx(1.0)  # the next value (episode start) is ignored


def test_policy_never_picks_masked_columns():
    torch.manual_seed(0)
    net = PolicyNet(input_dim=10, n_actions=4)
    x = torch.randn(500, 10)
    mask = torch.randint(0, 2, (500, 4), dtype=torch.int8)
    mask[:, 0] = 1  # at least one legal column per row
    for deterministic in (False, True):
        action, *_ = net.act(x, mask, deterministic=deterministic)
        assert mask[torch.arange(500), action].all()


def test_training_smoke_and_checkpoint_roundtrip(tables, tmp_path):
    cfg = PPOConfig(total_steps=256, n_envs=2, n_steps=16, n_minibatches=2, n_epochs=2, seed=1)
    envs = [ConnectFourOpponentEnv(tables, [0, 1, 2], tau=0.5) for _ in range(cfg.n_envs)]
    log = tmp_path / "log.csv"
    net = train_ppo(envs, cfg, log_path=log, verbose=False)

    assert net.input_dim == flat_obs_dim(tables.height, tables.width)
    assert log.exists() and len(log.read_text().splitlines()) == 1 + 256 // (2 * 16)
    assert all(torch.isfinite(p).all() for p in net.parameters())

    path = tmp_path / "policy.pt"
    save_policy(net, path)
    loaded = load_policy(path)
    x = torch.randn(3, net.input_dim)
    mask = torch.ones(3, net.n_actions, dtype=torch.int8)
    assert torch.allclose(net.distribution(x, mask).probs, loaded.distribution(x, mask).probs)


def test_policy_agent_plays_only_legal_moves(tables):
    env = ConnectFourOpponentEnv(tables, [4], tau=0.5)
    agent = PolicyAgent(PolicyNet(flat_obs_dim(env.height, env.width), env.width))
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action = agent(obs)
        assert obs["action_mask"][action] == 1
        obs, _, done, _, _ = env.step(action)
