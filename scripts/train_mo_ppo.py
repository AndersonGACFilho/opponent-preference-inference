"""Train the MO-PPO baseline (board + own omega, no opponent model) on the train opponents.

Example:
    uv run python scripts/train_mo_ppo.py --seed 0 --total-steps 2000000
Writes `policy.pt` and `train_log.csv` to results/mo_ppo/seed<seed>/ (git-ignored).
"""
import argparse
from pathlib import Path

import torch

from opponent_inference.agents.policy import save_policy
from opponent_inference.agents.ppo import PPOConfig, config_dict, train_ppo
from opponent_inference.env import ConnectFourOpponentEnv
from opponent_inference.solver import load_q_tables
from opponent_inference.splits import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--tau", type=float, default=0.1, help="opponent softmax temperature")
    parser.add_argument("--lr", type=float, default=PPOConfig.lr)
    parser.add_argument("--tables", type=Path, default=ROOT / "data" / "q_tables_4x4.npz")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)  # several seeds run in parallel, one core each
    out = args.out or ROOT / "results" / "mo_ppo" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    tables = load_q_tables(args.tables)
    train_ids, test_ids = train_test_split(tables)
    cfg = PPOConfig(total_steps=args.total_steps, lr=args.lr, seed=args.seed)
    envs = [ConnectFourOpponentEnv(tables, train_ids, tau=args.tau) for _ in range(cfg.n_envs)]
    print(f"train opponents {train_ids}, test opponents {test_ids}, tau {args.tau}", flush=True)

    net = train_ppo(envs, cfg, log_path=out / "train_log.csv")
    save_policy(
        net, out / "policy.pt",
        extra={"seed": args.seed, "tau": args.tau, "train_ids": train_ids, "test_ids": test_ids,
               "config": config_dict(cfg)},
    )
    print(f"saved {out / 'policy.pt'}")


if __name__ == "__main__":
    main()
