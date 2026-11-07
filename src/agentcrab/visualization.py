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
