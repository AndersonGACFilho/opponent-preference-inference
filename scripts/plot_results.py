"""Figures for the omega-hat agent vs MO-PPO: learning curves, test utility, heatmaps, inference.

Reads the training logs and evaluation CSVs written by `train_mo_ppo.py`, `train_belief_agent.py`
and `evaluate_agent.py`, plus `results/phase0_identification.csv`. Writes PNGs to
`results/figures/` (git-ignored).

Example:
    uv run python scripts/plot_results.py
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from opponent_inference.evaluate import summarize  # noqa: E402
from opponent_inference.solver import load_q_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AGENTS = {"mo_ppo": "MO-PPO", "belief": "ω̂ agent"}
COLORS = {"mo_ppo": "#7f8c8d", "belief": "#1f77b4"}
OUTCOMES = ("win_rate", "draw_rate", "loss_rate")


def learning_curves(results: Path, out: Path, window: int = 20) -> None:
    """Mean utility during training, averaged over seeds (band = min/max over seeds)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for agent, label in AGENTS.items():
        logs = []
        for path in sorted((results / agent).glob("seed*/train_log.csv")):
            log = pd.read_csv(path)
            log["mean_utility"] = log["mean_utility"].rolling(window, min_periods=1).mean()
            logs.append(log.set_index("global_step")["mean_utility"])
        curves = pd.concat(logs, axis=1)
        mean = curves.mean(axis=1)
        ax.plot(mean.index, mean.values, color=COLORS[agent], label=f"{label} (n={len(logs)})")
        ax.fill_between(mean.index, curves.min(axis=1), curves.max(axis=1),
                        color=COLORS[agent], alpha=0.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("passos de ambiente")
    ax.set_ylabel(f"utilidade média no treino (média móvel, {window} updates)")
    ax.set_title("Curvas de aprendizado (oponentes de treino)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def test_utility_by_opponent(df: pd.DataFrame, out: Path) -> None:
    """Mean utility on each test opponent with 95% CI over seeds."""
    stats = summarize(df, by=["agent", "opponent_id"])
    opponents = sorted(stats["opponent_id"].unique())
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, agent in enumerate(AGENTS):
        sub = stats[stats["agent"] == agent].set_index("opponent_id").loc[opponents]
        x = np.arange(len(opponents)) + (i - 0.5) * width
        ax.bar(x, sub["mean_utility_mean"], width, yerr=sub["mean_utility_ci95"], capsize=4,
               color=COLORS[agent], label=AGENTS[agent])
    ax.set_xticks(np.arange(len(opponents)))
    ax.set_xticklabels([f"oponente {k}" for k in opponents])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("utilidade média (ω·R)")
    ax.set_title("Utilidade no teste por oponente (IC 95% sobre seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def outcome_rates(df: pd.DataFrame, out: Path) -> None:
    """Win, draw and loss rates per agent, with 95% CI over seeds."""
    stats = summarize(df, by=["agent"]).set_index("agent").loc[list(AGENTS)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.36
    for i, outcome in enumerate(OUTCOMES):
        x = np.arange(len(AGENTS)) + (i - 1) * width / 2
        ax.bar(x, stats[f"{outcome}_mean"], width / 1.0,
               yerr=stats[f"{outcome}_ci95"], capsize=4,
               label=outcome.replace("_rate", "").replace("win", "vitória")
               .replace("draw", "empate").replace("loss", "derrota"))
    ax.set_xticks(np.arange(len(AGENTS)))
    ax.set_xticklabels(list(AGENTS.values()))
    ax.set_ylabel("fração de jogos")
    ax.set_title("Resultados no teste (IC 95% sobre seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def utility_heatmaps(df: pd.DataFrame, weights: np.ndarray, out: Path) -> None:
    """Utility per (own omega, test opponent): MO-PPO, omega-hat agent, and their difference."""
    cell = df.groupby(["agent", "own_omega_id", "opponent_id"])["mean_utility"].mean().unstack(
        "opponent_id")
    grids = {agent: cell.loc[agent] for agent in AGENTS}
    diff = grids["belief"] - grids["mo_ppo"]
    labels = [f"({w[0]:.2f},{w[1]:.2f},{w[2]:.2f})" for w in weights]
    opponents = list(diff.columns)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5))
    vmax = max(abs(grids["mo_ppo"].values).max(), abs(grids["belief"].values).max())
    panels = [
        (grids["mo_ppo"], "MO-PPO", "RdBu", -vmax, vmax),
        (grids["belief"], "ω̂ agent", "RdBu", -vmax, vmax),
        (diff, "ω̂ − MO-PPO", "PiYG", -np.abs(diff.values).max(), np.abs(diff.values).max()),
    ]
    for ax, (grid, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(grid.values, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(opponents)))
        ax.set_xticklabels([f"opp {k}" for k in opponents])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ax is axes[0] else [""] * len(labels), fontsize=8)
        ax.set_xlabel("oponente de teste")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("ω próprio do agente")
    fig.suptitle("Utilidade por ω próprio e oponente de teste (média sobre lados e seeds)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def inference_quality(phase0: Path, out: Path) -> None:
    """Posterior mass on the true behaviour group after 1, 2, 4 and 8 opponent moves."""
    df = pd.read_csv(phase0)
    moves = [1, 2, 4, 8]
    cols = [f"mass_after_{m}" for m in moves]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for _, row in df.iterrows():
        ax.plot(moves, row[cols].values.astype(float), color="#bbbbbb", lw=1)
    ax.plot(moves, df[cols].astype(float).mean(), color="#1f77b4", lw=2.5,
            label="média dos 15 oponentes")
    ax.axhline(0.9, color="red", ls="--", label="gate da fase 0 (0,9)")
    ax.set_xlabel("lances do oponente observados")
    ax.set_ylabel("massa posterior no grupo verdadeiro")
    ax.set_ylim(0, 1)
    ax.set_xticks(moves)
    ax.set_title("Identificação do oponente pelo filtro (agente aleatório, τ = 0,1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--tables", type=Path, default=ROOT / "data" / "q_tables_4x4.npz")
    args = parser.parse_args()

    out_dir = args.results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = load_q_tables(args.tables)

    eval_df = pd.concat([
        pd.read_csv(args.results / "mo_ppo" / "eval_test.csv"),
        pd.read_csv(args.results / "belief" / "eval_test.csv"),
    ], ignore_index=True)
    eval_df = eval_df[eval_df["agent"].isin(AGENTS)]

    learning_curves(args.results, out_dir / "learning_curves.png")
    test_utility_by_opponent(eval_df, out_dir / "test_utility_by_opponent.png")
    outcome_rates(eval_df, out_dir / "test_outcome_rates.png")
    utility_heatmaps(eval_df, tables.weights, out_dir / "utility_heatmaps.png")
    inference_quality(args.results / "phase0_identification.csv",
                      out_dir / "inference_quality.png")
    for path in sorted(out_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
