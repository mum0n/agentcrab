"""Analysis and statistical models for AgentCrab."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ── Stock-recruitment models ──────────────────────────────────────────


def schaefer_model(spawners: np.ndarray, r: float, K: float) -> np.ndarray:
    """Schaefer surplus production model: R = r*S*(1 - S/K)."""
    return r * spawners * (1.0 - spawners / K)


def fit_schaefer(spawners: np.ndarray, recruits: np.ndarray) -> dict:
    """Fit Schaefer stock-recruitment model."""
    try:
        popt, pcov = curve_fit(schaefer_model, spawners, recruits, p0=[0.5, 1000.0], maxfev=10000)
        r, K = popt
        residuals = recruits - schaefer_model(spawners, r, K)
        rmse = np.sqrt(np.mean(residuals**2))
        return {"r": float(r), "K": float(K), "rmse": float(rmse), "model": "schaefer"}
    except Exception as e:
        return {"r": 0.0, "K": 0.0, "rmse": float("inf"), "error": str(e)}


def delay_difference_step(b: float, n: float, y: float, r: float, cfg) -> Tuple[float, float]:
    """Single step of Deriso-Schnute delay-difference model."""
    s = cfg.get("s", 0.8)
    alpha = cfg.get("alpha", 12.0)
    rho = cfg.get("rho", 0.3)
    wk = cfg.get("wk", 2.0)
    b_esc = max(0.0, b - y)
    mean_w = n if n > 0 else wk
    ct = y / mean_w if mean_w > 0 else 0.0
    n_esc = max(0.0, n - ct)
    next_n = s * n_esc + r
    next_b = s * alpha * n_esc + s * rho * b_esc + wk * r
    return max(0.0, next_b), max(0.0, next_n)


def fit_delay_difference(biomass: np.ndarray, catch: np.ndarray, recruits: np.ndarray) -> dict:
    """Fit delay-difference model (simplified: returns parameters)."""
    cfg = {"s": 0.8, "alpha": 12.0, "rho": 0.3, "wk": 2.0}
    ny = len(biomass)
    b = np.zeros(ny)
    n = np.zeros(ny)
    b[0] = biomass[0]
    n[0] = biomass[0] / cfg["wk"]
    for t in range(ny - 1):
        nb, nn = delay_difference_step(b[t], n[t], catch[t], recruits[t + 1] if t + 1 < len(recruits) else 0.0, cfg)
        b[t + 1] = nb
        n[t + 1] = nn
    rmse = np.sqrt(np.mean((biomass - b) ** 2))
    return {"cfg": cfg, "rmse": float(rmse), "model": "delay_difference", "biomass_fit": b}


# ── Leslie matrix model ──────────────────────────────────────────────


def build_leslie_matrix(params, n_stages: int = 5) -> np.ndarray:
    """Build a Leslie matrix from parameters."""
    L = np.zeros((n_stages, n_stages))
    # Survival on subdiagonal
    for i in range(1, n_stages):
        L[i, i - 1] = params.leslie.survival_stage1 if i == 1 else params.leslie.survival_stage2
    # Fecundity on first row
    for j in range(n_stages):
        L[0, j] = params.leslie.fecundity_base * (params.leslie.fecundity_size_reference / 100.0) ** params.leslie.fecundity_size_exponent
    return L


def leslie_eigenvalues(L: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of Leslie matrix."""
    return np.linalg.eigvals(L)


def summarize_leslie_model(L: np.ndarray) -> dict:
    """Summarize Leslie matrix: growth rate, stable age distribution."""
    eigs = leslie_eigenvalues(L)
    # Dominant eigenvalue (largest real part)
    idx = np.argmax(np.real(eigs))
    lambda1 = eigs[idx]
    r = np.log(np.abs(lambda1)) if np.abs(lambda1) > 0 else 0.0
    # Stable age distribution (right eigenvector)
    w, v = np.linalg.eig(L)
    idx2 = np.argmax(np.real(w))
    sad = np.abs(v[:, idx2])
    sad = sad / sad.sum()
    return {
        "dominant_eigenvalue": complex(lambda1),
        "growth_rate_r": float(r),
        "stable_age_distribution": sad.tolist(),
    }


# ── Population criteria / evaluation ─────────────────────────────────


def evaluate_population_criteria(model_df, params) -> dict:
    """Evaluate multi-attribute population criteria."""
    crit = params.criteria
    results = {}

    # Persistence
    if len(model_df) > 0 and "n_agents" in model_df.columns:
        min_pop = model_df["n_agents"].min()
        results["persistence"] = min_pop >= crit.min_population
        results["min_population"] = int(min_pop)

    # Explosion check
    if len(model_df) > 0 and "n_agents" in model_df.columns:
        max_pop = model_df["n_agents"].max()
        k = params.environment.benthic_carrying_capacity
        results["no_explosion"] = max_pop <= k * crit.max_population_multiple
        results["max_population"] = int(max_pop)

    # Size drift
    if "mean_cw" in model_df.columns and len(model_df) > crit.window_steps:
        recent = model_df["mean_cw"].tail(crit.window_steps)
        drift = (recent.iloc[-1] - recent.iloc[0]) / max(1.0, recent.iloc[0])
        results["size_drift_ok"] = abs(drift) <= crit.max_size_drift
        results["size_drift"] = float(drift)

    return results


# ── Surplus production models ─────────────────────────────────────────


def schaefer_surplus_production(B: np.ndarray, r: float, K: float, q: float = 1.0) -> np.ndarray:
    """Schaefer surplus production: P = r*B*(1 - B/K)."""
    return r * B * (1.0 - B / K)


def fox_surplus_production(B: np.ndarray, r: float, K: float) -> np.ndarray:
    """Fox (Gompertz) surplus production: P = r*B*ln(K/B)."""
    return r * B * np.log(K / np.clip(B, 1e-6, None))


def pella_tomlinson_surplus_production(B: np.ndarray, r: float, K: float, m: float = 2.0) -> np.ndarray:
    """Pella-Tomlinson surplus production: P = r*B*(1 - (B/K)^m)/m."""
    return r * B * (1.0 - (B / K) ** m) / m


# ── Mohn's rho (retrospective analysis) ──────────────────────────────


def mohn_rho(estimates: np.ndarray) -> float:
    """Calculate Mohn's rho for retrospective analysis.

    rho = (1/n) * sum((est_i - est_full) / est_full)
    """
    if len(estimates) < 2:
        return 0.0
    full = estimates[-1]
    if full == 0:
        return 0.0
    rho = np.mean((estimates[:-1] - full) / full)
    return float(rho)


# ── Cohort / aggregate group helpers ──────────────────────────────────


def individuate_cohort(group: dict, params) -> list:
    """Individuate an aggregate cohort into individual organisms.

    Returns a list of organism dicts.
    """
    organisms = []
    n = group.get("count", 0)
    mean_size = group.get("mean_size", 10.0)
    for i in range(n):
        organisms.append({
            "stage": group.get("stage", "juvenile"),
            "carapace_width": mean_size,
            "age_days": group.get("age_days", 0.0),
            "sex": "male" if i % 2 == 0 else "female",
        })
    return organisms
