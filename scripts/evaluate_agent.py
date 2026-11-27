"""Evaluate trained policies (or a random agent) against fixed opponents.

Examples:
    uv run python scripts/evaluate_agent.py --checkpoints results/mo_ppo/seed*/policy.pt
    uv run python scripts/evaluate_agent.py --random --opponents test
Writes a per-cell CSV (own omega x opponent x side x seed) and prints a summary over seeds.
"""
import argparse
from functools import partial
from pathlib import Path

import pandas as pd
import torch

from opponent_inference.agents.policy import PolicyAgent, load_policy
from opponent_inference.evaluate import evaluate_policy, random_policy, summarize
from opponent_inference.inference import BeliefWrapper, belief_obs_fn
from opponent_inference.solver import load_q_tables
from opponent_inference.splits import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=[])
    parser.add_argument("--random", action="store_true", help="also evaluate a random agent")
    parser.add_argument("--opponents", choices=["train", "test", "all"], default="test")
    parser.add_argument("--games", type=int, default=100, help="games per (omega, opponent, side)")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--tables", type=Path, default=ROOT / "data" / "q_tables_4x4.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "eval.csv")
    args = parser.parse_args()

    torch.set_num_threads(1)
    tables = load_q_tables(args.tables)
    train_ids, test_ids = train_test_split(tables)
    opponent_ids = {"train": train_ids, "test": test_ids, "all": list(range(len(tables.weights)))}[
        args.opponents
    ]
    own_omega_ids = list(range(len(tables.weights)))

    frames = []
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        extra = ckpt["extra"]
        seed = int(extra.get("seed", 0))
        if extra.get("agent") == "belief":  # omega-hat agent: policy input includes the belief
            name = "belief"
            agent = PolicyAgent(load_policy(path), obs_fn=belief_obs_fn)
            hypotheses = extra["hypothesis_ids"]
            env_wrapper = partial(BeliefWrapper, tables=tables, hypothesis_ids=hypotheses,
                                  tau=args.tau)
        else:
            name = "mo_ppo"
            agent = PolicyAgent(load_policy(path))
            env_wrapper = None
        df = evaluate_policy(agent, tables, opponent_ids, own_omega_ids, args.games, args.tau,
                             seed, env_wrapper=env_wrapper)
        frames.append(df.assign(agent=name, seed=seed))
        print(f"evaluated {path} ({name}, seed {seed})", flush=True)
    if args.random:
        df = evaluate_policy(random_policy(0), tables, opponent_ids, own_omega_ids,
                             args.games, args.tau, seed=0)
        frames.append(df.assign(agent="random", seed=0))

    if not frames:
        parser.error("nothing to evaluate: pass --checkpoints and/or --random")
    result = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    for agent, group in result.groupby("agent"):
        print(f"\n== {agent} vs {args.opponents} opponents {opponent_ids}")
        print(summarize(group).to_string(index=False))


if __name__ == "__main__":
    main()
