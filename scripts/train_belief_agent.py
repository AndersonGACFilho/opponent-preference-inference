"""Train the omega-hat agent: best response to the inferred opponent omega (belief input).

The agent sees the board, its own omega and the posterior of a Bayesian filter over the train
opponents' grid indices. The filter is updated by the opponent's moves inside each episode.

Example:
    uv run python scripts/train_belief_agent.py --seed 0 --total-steps 2000000
Writes `policy.pt` and `train_log.csv` to results/belief/seed<seed>/ (git-ignored).
"""
import argparse
from pathlib import Path

import torch

from opponent_inference.agents.policy import save_policy
from opponent_inference.agents.ppo import PPOConfig, config_dict, train_ppo
from opponent_inference.env import ConnectFourOpponentEnv
from opponent_inference.inference import BeliefWrapper, belief_obs_fn
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
    out = args.out or ROOT / "results" / "belief" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    tables = load_q_tables(args.tables)
    train_ids, test_ids = train_test_split(tables)
    cfg = PPOConfig(total_steps=args.total_steps, lr=args.lr, seed=args.seed)
    # Hypotheses = train opponents (decision D1): test opponents never enter training or inference.
    envs = [
        BeliefWrapper(
            ConnectFourOpponentEnv(tables, train_ids, tau=args.tau), tables, train_ids, args.tau
        )
        for _ in range(cfg.n_envs)
    ]
    print(f"train opponents {train_ids}, test opponents {test_ids}, tau {args.tau}", flush=True)

    net = train_ppo(envs, cfg, obs_fn=belief_obs_fn, log_path=out / "train_log.csv")
    save_policy(
        net, out / "policy.pt",
        extra={"agent": "belief", "seed": args.seed, "tau": args.tau,
               "train_ids": train_ids, "test_ids": test_ids,
               "hypothesis_ids": train_ids, "config": config_dict(cfg)},
    )
    print(f"saved {out / 'policy.pt'}")


if __name__ == "__main__":
    main()
