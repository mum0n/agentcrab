"""Generate visualization plots from saved simulation results."""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot AgentCrab simulation results")
    parser.add_argument("--input", default="output_5yr", help="Directory with Parquet results")
    parser.add_argument("--output", default=None, help="Output directory for plots (default: same as input)")
    args = parser.parse_args(argv)

    out_dir = args.output or args.input
    os.makedirs(out_dir, exist_ok=True)

    # Load results
    model_df = pd.read_parquet(os.path.join(args.input, "model_results.parquet"))
    agent_df = pd.read_parquet(os.path.join(args.input, "agent_results.parquet"))

    # 1. Population timeseries
    fig, ax = plt.subplots(figsize=(12, 6))
    if "n_agents" in model_df.columns:
        ax.plot(model_df.index, model_df["n_agents"], "b-", lw=2, label="Total")
    if "n_mature" in model_df.columns:
        ax.plot(model_df.index, model_df["n_mature"], "g--", lw=1.5, label="Mature")
    if "n_legal" in model_df.columns:
        ax.plot(model_df.index, model_df["n_legal"], "r:", lw=1.5, label="Legal")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Population Abundance Over Time (5-Year Snow Crab)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "timeseries.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/timeseries.png")

    # 2. Stage distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    if "stage" in agent_df.columns:
        stage_counts = agent_df["stage"].value_counts()
        stage_counts.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_xlabel("Life Stage")
        ax.set_ylabel("Count")
        ax.set_title("Stage Distribution")
        ax.tick_params(axis="x", rotation=45)
    fig.savefig(os.path.join(out_dir, "stage_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/stage_distribution.png")

    # 3. Size by sex
    fig, ax = plt.subplots(figsize=(8, 5))
    if "carapace_width" in agent_df.columns and "sex" in agent_df.columns:
        for sex, color in [("male", "blue"), ("female", "red")]:
            subset = agent_df[agent_df["sex"] == sex]["carapace_width"]
            if len(subset) > 0:
                ax.hist(subset, bins=30, alpha=0.5, color=color, label=sex, density=True)
        ax.set_xlabel("Carapace Width (mm)")
        ax.set_ylabel("Density")
        ax.set_title("Size Distribution by Sex")
        ax.legend()
    fig.savefig(os.path.join(out_dir, "size_by_sex.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/size_by_sex.png")

    # 4. Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(model_df.index, model_df["n_agents"], "b-", lw=2)
    axes[0, 0].set_title("Population Abundance")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("N")
    axes[0, 0].grid(True, alpha=0.3)

    if "stage" in agent_df.columns:
        stage_counts = agent_df["stage"].value_counts()
        stage_counts.plot(kind="bar", ax=axes[0, 1], color="steelblue")
        axes[0, 1].set_title("Stage Distribution")
        axes[0, 1].tick_params(axis="x", rotation=45)

    if "carapace_width" in agent_df.columns and "sex" in agent_df.columns:
        for sex, color in [("male", "blue"), ("female", "red")]:
            subset = agent_df[agent_df["sex"] == sex]["carapace_width"]
            if len(subset) > 0:
                axes[1, 0].hist(subset, bins=30, alpha=0.5, color=color, label=sex, density=True)
        axes[1, 0].set_title("Size Distribution by Sex")
        axes[1, 0].set_xlabel("Carapace Width (mm)")
        axes[1, 0].legend()

    # Mean CW over time
    if "mean_cw" in model_df.columns:
        axes[1, 1].plot(model_df.index, model_df["mean_cw"], "g-", lw=2)
    axes[1, 1].set_title("Mean Carapace Width Over Time")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("CW (mm)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/dashboard.png")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
