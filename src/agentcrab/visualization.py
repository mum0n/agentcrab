"""Visualization module for AgentCrab using matplotlib."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .simulation import CrabModel


def plot_population_timeseries(model_df: pd.DataFrame, save_path: str = None):
    """Plot population abundance over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(model_df.index, model_df["n_agents"], "b-", lw=2, label="Total")
    if "n_mature" in model_df.columns:
        ax.plot(model_df.index, model_df["n_mature"], "g--", lw=1.5, label="Mature")
    if "n_legal" in model_df.columns:
        ax.plot(model_df.index, model_df["n_legal"], "r:", lw=1.5, label="Legal")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Population Abundance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_stage_distribution(agent_df: pd.DataFrame, save_path: str = None):
    """Plot stage distribution as a bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if "stage" in agent_df.columns:
        stage_counts = agent_df["stage"].value_counts()
        stage_counts.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_xlabel("Life Stage")
        ax.set_ylabel("Count")
        ax.set_title("Stage Distribution")
        ax.tick_params(axis="x", rotation=45)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_size_by_sex(agent_df: pd.DataFrame, save_path: str = None):
    """Plot size distribution by sex."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if "carapace_width" in agent_df.columns and "sex" in agent_df.columns:
        for sex, color in [("male", "blue"), ("female", "red")]:
            subset = agent_df[agent_df["sex"] == sex]["carapace_width"]
            if len(subset) > 0:
                ax.hist(subset, bins=20, alpha=0.5, color=color, label=sex, density=True)
        ax.set_xlabel("Carapace Width (mm)")
        ax.set_ylabel("Density")
        ax.set_title("Size Distribution by Sex")
        ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_spatial_habitat(model, save_path: str = None):
    """Plot spatial distribution of agents on the grid."""
    fig, ax = plt.subplots(figsize=(8, 8))
    positions = [(a.pos[0], a.pos[1]) for a in model.agents]
    if positions:
        xs, ys = zip(*positions)
        ax.scatter(xs, ys, c="blue", s=5, alpha=0.5)
    ax.set_xlim(0, model.grid_size[0])
    ax.set_ylim(0, model.grid_size[1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Spatial Distribution of Agents")
    ax.set_aspect("equal")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_simulation_dashboard(model, model_df, agent_df, save_path: str = None):
    """Create a 2x2 dashboard of simulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Population timeseries
    axes[0, 0].plot(model_df.index, model_df["n_agents"], "b-", lw=2)
    axes[0, 0].set_title("Population Abundance")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("N")
    axes[0, 0].grid(True, alpha=0.3)

    # Stage distribution
    if "stage" in agent_df.columns:
        stage_counts = agent_df["stage"].value_counts()
        stage_counts.plot(kind="bar", ax=axes[0, 1], color="steelblue")
        axes[0, 1].set_title("Stage Distribution")
        axes[0, 1].tick_params(axis="x", rotation=45)

    # Size by sex
    if "carapace_width" in agent_df.columns and "sex" in agent_df.columns:
        for sex, color in [("male", "blue"), ("female", "red")]:
            subset = agent_df[agent_df["sex"] == sex]["carapace_width"]
            if len(subset) > 0:
                axes[1, 0].hist(subset, bins=20, alpha=0.5, color=color, label=sex, density=True)
        axes[1, 0].set_title("Size Distribution by Sex")
        axes[1, 0].set_xlabel("Carapace Width (mm)")
        axes[1, 0].legend()

    # Spatial distribution
    positions = [(a.pos[0], a.pos[1]) for a in model.agents]
    if positions:
        xs, ys = zip(*positions)
        axes[1, 1].scatter(xs, ys, c="blue", s=5, alpha=0.5)
    axes[1, 1].set_title("Spatial Distribution")
    axes[1, 1].set_xlim(0, model.grid_size[0])
    axes[1, 1].set_ylim(0, model.grid_size[1])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def fmt_param(val, digits: int = 2) -> str:
    return f"{float(val):.{digits}f}" if isinstance(val, (int, float)) else str(val)


def organism_color(org) -> str:
    if org.sex.value == "female":
        return "#ef4444" if org.is_mature else "#f97316"
    if org.is_mature:
        return "#2563eb"
    return "#38bdf8"


def organism_size(org) -> float:
    return max(3.0, min(14.0, org.carapace_width / 12.0))


def organism_marker(org) -> str:
    return "o" if org.sex.value == "female" else "^"


def plot_stage_allometry(agent_df, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(agent_df) > 0 and "carapace_width" in agent_df.columns and "mass" in agent_df.columns:
        ax.scatter(agent_df["carapace_width"], agent_df["mass"], s=8, alpha=0.4)
    ax.set_title("Stage & Size Allometry")
    ax.set_xlabel("Carapace Width (mm)")
    ax.set_ylabel("Wet Mass (g)")
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_reproduction_dynamics(model_df, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    if "n_mature_females" in model_df.columns:
        ax.plot(model_df.index, model_df["n_mature_females"], label="Mature Females", color="#ef4444")
    if "n_ovigerous" in model_df.columns:
        ax.plot(model_df.index, model_df["n_ovigerous"], label="Ovigerous", color="#f59e0b")
    ax.set_title("Reproduction Dynamics")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_maturity_ogives(agent_df, params, save_path: Optional[str] = None):
    from .species import maturity_probability
    from .types import Sex

    fig, ax = plt.subplots(figsize=(8, 5))
    cws = np.linspace(10.0, 150.0, 140)
    ax.plot(cws, [maturity_probability(x, Sex.MALE, params) for x in cws], label="Male (Theory)", color="#2563eb")
    ax.plot(cws, [maturity_probability(x, Sex.FEMALE, params) for x in cws], label="Female (Theory)", color="#ef4444")
    ax.set_title("Maturity Ogives")
    ax.set_xlabel("Carapace Width (mm)")
    ax.set_ylabel("Proportion Mature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_stock_recruitment(spawners, recruits, sr_fit: Optional[dict] = None, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(spawners) > 0:
        ax.scatter(spawners, recruits, color="#0f172a", label="Observed")
    ax.set_title("Stock-Recruitment")
    ax.set_xlabel("Spawning Stock")
    ax.set_ylabel("Recruits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_population_dynamics_fits(observed, comp: dict, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(observed, "o", color="#0f172a", label="Observed")
    for k, col in [("schaefer", "#0d9488"), ("delay_difference", "#8b5cf6")]:
        if k in comp and comp[k] and "predicted" in comp[k]:
            ax.plot(comp[k]["predicted"], color=col, label=f"{k} (R2={comp[k].get('r2', 0):.2f})")
    ax.set_title("Macro-Ecological Model Fits")
    ax.set_xlabel("Year")
    ax.set_ylabel("Abundance / Biomass")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_macro_model_comparison(observed, comp: dict, save_path: Optional[str] = None):
    return plot_population_dynamics_fits(observed, comp, save_path=save_path)


def plot_retrospective_analysis(retro: dict, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    pts = retro.get("peel_endpoints", [])
    if pts:
        ax.plot(range(1, len(pts) + 1), pts, "-o", color="#dc2626")
    ax.set_title(f"Retrospective Analysis (Mohn's rho={retro.get('mohn_rho', 0.0):.3f})")
    ax.set_xlabel("Peel")
    ax.set_ylabel("Terminal Estimate")
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_stage_specific_dynamics(agent_df, save_path: Optional[str] = None):
    return plot_stage_distribution(agent_df, save_path=save_path)


def plot_abm_summary(model, model_df, agent_df, save_path: Optional[str] = None):
    return plot_simulation_dashboard(model, model_df, agent_df, save_path=save_path)


def plot_functional_relationships(params, save_path: Optional[str] = None):
    return plot_maturity_ogives(None, params, save_path=save_path)


def export_all_plots(model, model_df, agent_df, output_dir: str = "results") -> dict:
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "dashboard": str(out / "dashboard.png"),
        "timeseries": str(out / "population_timeseries.png"),
        "allometry": str(out / "stage_allometry.png"),
        "ogives": str(out / "maturity_ogives.png"),
    }
    plot_simulation_dashboard(model, model_df, agent_df, save_path=paths["dashboard"])
    plot_population_timeseries(model_df, save_path=paths["timeseries"])
    plot_stage_allometry(agent_df, save_path=paths["allometry"])
    plot_maturity_ogives(agent_df, model.params, save_path=paths["ogives"])
    plt.close("all")
    return paths
