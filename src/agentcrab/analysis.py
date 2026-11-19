"""Analysis, macro-ecological assessment, and calibration models for AgentCrab."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit, minimize


# ── Stock-recruitment models ──────────────────────────────────────────


def schaefer_model(spawners: np.ndarray, r: float, K: float) -> np.ndarray:
    """Schaefer surplus production function: r*S*(1 - S/K)."""
    return r * spawners * (1.0 - spawners / max(1e-9, K))


def fit_stock_recruitment(spawners: np.ndarray, recruits: np.ndarray) -> dict:
    """Fit Beverton-Holt (R = a*S/(1 + b*S)) and Ricker (R = a*S*exp(-b*S)) models."""
    s = np.asarray(spawners, dtype=float)
    r = np.asarray(recruits, dtype=float)
    mask = np.isfinite(s) & np.isfinite(r) & (s > 0) & (r >= 0)
    s, r = s[mask], r[mask]
    if len(s) < 3:
        return {"beverton_holt": None, "ricker": None}

    n = len(s)
    ss_tot = float(np.sum((r - np.mean(r)) ** 2)) + 1e-12

    def _fit_one(fn, p0):
        res = minimize(
            lambda p: float(np.sum((r - fn(s, math.exp(p[0]), math.exp(p[1]))) ** 2)),
            [math.log(max(1e-4, p0[0])), math.log(max(1e-6, p0[1]))],
            method="Nelder-Mead",
        )
        a, b = math.exp(res.x[0]), math.exp(res.x[1])
        pred = fn(s, a, b)
        rss = float(np.sum((r - pred) ** 2))
        rmse = math.sqrt(rss / n)
        r2 = 1.0 - rss / ss_tot
        aic = n * math.log(max(1e-12, rss / n)) + 4.0
        return {"alpha": a, "beta": b, "rmse": rmse, "R2": r2, "aic": aic, "predicted": pred}

    a0 = float(np.max(r) / max(1e-6, np.mean(s)))
    b0 = 1.0 / max(1e-6, np.mean(s))
    return {
        "beverton_holt": _fit_one(lambda x, a, b: (a * x) / (1.0 + b * x), (a0, b0)),
        "ricker": _fit_one(lambda x, a, b: a * x * np.exp(-b * x), (a0, b0)),
    }


def _simulate_schaefer(n0: float, r: float, k: float, catch: np.ndarray) -> np.ndarray:
    n_years = len(catch)
    pred = np.zeros(n_years, dtype=float)
    pred[0] = max(1e-6, n0)
    for t in range(n_years - 1):
        nt = pred[t]
        prod = r * nt * (1.0 - nt / max(1e-9, k))
        pred[t + 1] = max(1e-6, nt + prod - catch[t])
    return pred


def fit_schaefer(
    observed: np.ndarray,
    catch_or_recruits: Optional[np.ndarray] = None,
    dynamic: Optional[bool] = None,
) -> dict:
    """Fit dynamic Schaefer surplus-production model (or static S-R when dynamic=False)."""
    obs = np.asarray(observed, dtype=float)
    # Backward compatibility: 2-arg (spawners, recruits) static fit when values look like S-R
    if dynamic is False:
        rec = np.asarray(catch_or_recruits, dtype=float)
        popt, _ = curve_fit(schaefer_model, obs, rec, p0=[0.5, max(1.0, float(np.max(obs)) * 2.0)], maxfev=10000)
        r, k = float(popt[0]), float(popt[1])
        rmse = float(np.sqrt(np.mean((rec - schaefer_model(obs, r, k)) ** 2)))
        return {"r": r, "K": k, "rmse": rmse, "model": "schaefer"}

    catch = np.zeros_like(obs) if catch_or_recruits is None else np.asarray(catch_or_recruits, dtype=float)
    if len(obs) < 3:
        return {"r": 0.0, "K": 0.0, "MSY": 0.0, "R2": 0.0, "rmse": float("inf"), "aic": float("inf"), "model": "schaefer"}

    scale = max(1e-6, float(np.mean(obs)))
    obs_s = obs / scale
    catch_s = catch / scale
    ss_tot = float(np.sum((obs_s - np.mean(obs_s)) ** 2)) + 1e-12

    best = None
    for r0 in (0.2, 0.45, 0.8):
        for k0 in (1.2, 2.0, 3.5):
            def obj(theta):
                r_val = 0.01 + 2.5 / (1.0 + math.exp(-theta[0]))
                k_val = max(0.2, math.exp(theta[1]))
                n0_val = max(0.05, math.exp(theta[2]))
                p = _simulate_schaefer(n0_val, r_val, k_val, catch_s)
                return float(np.sum((obs_s - p) ** 2))

            x0 = [math.log(r0 / max(1e-4, 2.5 - r0)), math.log(k0), math.log(max(0.1, obs_s[0]))]
            res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 600})
            if best is None or res.fun < best.fun:
                best = res

    r_hat = 0.01 + 2.5 / (1.0 + math.exp(-best.x[0]))
    k_hat = math.exp(best.x[1]) * scale
    n0_hat = math.exp(best.x[2]) * scale
    pred = _simulate_schaefer(n0_hat, r_hat, k_hat, catch)
    rss = float(np.sum((obs - pred) ** 2))
    n = len(obs)
    rmse = math.sqrt(rss / n)
    r2 = 1.0 - float(np.sum((obs_s - pred / scale) ** 2)) / ss_tot
    aic = n * math.log(max(1e-12, rss / n)) + 6.0
    return {
        "r": float(r_hat),
        "K": float(k_hat),
        "n0": float(n0_hat),
        "MSY": float(r_hat * k_hat / 4.0),
        "N_msy": float(k_hat / 2.0),
        "U_msy": float(r_hat / 2.0),
        "R2": float(r2),
        "rmse": float(rmse),
        "aic": float(aic),
        "predicted": pred,
        "model": "schaefer",
    }


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


def _simulate_delay_difference(
    n0: float, s: float, alpha: float, beta_rel: float, catch_s: np.ndarray, lag: int = 1
) -> np.ndarray:
    ny = len(catch_s)
    pred = np.zeros(ny, dtype=float)
    pred[0] = max(1e-6, n0)
    for t in range(ny - 1):
        esc = max(0.0, pred[t] - catch_s[t])
        idx_lag = max(0, t + 1 - lag)
        spawners = pred[idx_lag]
        rec = (alpha * spawners) / (1.0 + beta_rel * spawners)
        pred[t + 1] = max(1e-6, s * esc + rec)
    return pred


def fit_delay_difference(
    biomass: np.ndarray,
    catch: Optional[np.ndarray] = None,
    recruits: Optional[np.ndarray] = None,
    lag: int = 1,
) -> dict:
    """Fit dynamic Beverton-Holt delay-difference model matching TethysBio."""
    obs = np.asarray(biomass, dtype=float)
    c = np.zeros_like(obs) if catch is None else np.asarray(catch, dtype=float)
    if len(obs) < 3:
        return {"s": 0.0, "M": 0.0, "alpha": 0.0, "beta": 0.0, "rmse": float("inf"), "R2": 0.0, "aic": float("inf"), "biomass_fit": obs.copy(), "model": "delay_difference"}

    scale = max(1e-6, float(np.mean(obs)))
    obs_s = obs / scale
    c_s = c / scale
    ss_tot = float(np.sum((obs_s - np.mean(obs_s)) ** 2)) + 1e-12

    best = None
    for s0 in (0.55, 0.78):
        for a0 in (0.4, 1.2):
            def obj(theta):
                s_val = 0.05 + 0.93 / (1.0 + math.exp(-theta[0]))
                a_val = math.exp(theta[1])
                b_val = math.exp(theta[2])
                p = _simulate_delay_difference(obs_s[0], s_val, a_val, b_val, c_s, lag=lag)
                return float(np.sum((obs_s - p) ** 2))

            x0 = [math.log(s0 / (0.98 - s0)), math.log(a0), math.log(0.5)]
            res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 600})
            if best is None or res.fun < best.fun:
                best = res

    s_hat = 0.05 + 0.93 / (1.0 + math.exp(-best.x[0]))
    a_hat = math.exp(best.x[1])
    b_rel = math.exp(best.x[2])
    pred = _simulate_delay_difference(obs_s[0], s_hat, a_hat, b_rel, c_s, lag=lag) * scale
    rss = float(np.sum((obs - pred) ** 2))
    n = len(obs)
    rmse = math.sqrt(rss / n)
    r2 = 1.0 - float(np.sum((obs_s - pred / scale) ** 2)) / ss_tot
    aic = n * math.log(max(1e-12, rss / n)) + 6.0
    return {
        "s": float(s_hat),
        "M": float(-math.log(max(1e-6, s_hat))),
        "alpha": float(a_hat),
        "beta": float(b_rel / scale),
        "cfg": {"s": float(s_hat), "alpha": float(a_hat), "beta": float(b_rel / scale)},
        "R2": float(r2),
        "rmse": float(rmse),
        "aic": float(aic),
        "biomass_fit": pred,
        "predicted": pred,
        "model": "delay_difference",
    }


# ── Leslie matrix model ──────────────────────────────────────────────


def build_leslie_matrix(params, n_stages: int = 5) -> np.ndarray:
    """Build a Leslie matrix from parameters."""
    L = np.zeros((n_stages, n_stages))
    for i in range(1, n_stages):
        L[i, i - 1] = params.leslie.survival_stage1 if i == 1 else params.leslie.survival_stage2
    for j in range(n_stages):
        L[0, j] = params.leslie.fecundity_base * (params.leslie.fecundity_size_reference / 100.0) ** params.leslie.fecundity_size_exponent
    return L


def leslie_eigenvalues(L: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(L)


def summarize_leslie_model(L: np.ndarray) -> dict:
    """Summarize Leslie matrix: dominant eigenvalue, growth rate, SSD, damping ratio."""
    eigs = np.linalg.eigvals(L)
    mags = np.abs(eigs)
    order = np.argsort(-mags)
    lambda1 = eigs[order[0]]
    lambda2 = eigs[order[1]] if len(order) > 1 else 0.0
    r = math.log(max(1e-12, float(np.real(lambda1))))
    w, v = np.linalg.eig(L)
    idx2 = int(np.argmax(np.real(w)))
    sad = np.abs(v[:, idx2])
    sad = sad / max(1e-12, float(sad.sum()))
    damping = float(abs(lambda1) / max(1e-12, abs(lambda2))) if abs(lambda2) > 0 else float("inf")
    return {
        "dominant_eigenvalue": complex(lambda1),
        "lambda1": float(np.real(lambda1)),
        "growth_rate_r": float(r),
        "damping_ratio": damping,
        "stable_age_distribution": sad.tolist(),
    }


def fit_generalized_leslie_model(stage_matrix: np.ndarray, params=None) -> dict:
    """Fit a stage-structured generalized Leslie transition matrix to annual stage counts."""
    mat = np.asarray(stage_matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[1] < 2:
        L = build_leslie_matrix(params) if params is not None else np.eye(5)
        s = summarize_leslie_model(L)
        s.update({"L": L, "R2": 0.0, "rmse": float("inf"), "aic": float("inf"), "model": "leslie"})
        return s

    n_st, n_yr = mat.shape
    obs_tot = mat.sum(axis=0)
    scale = max(1e-6, float(np.mean(obs_tot)))

    def _build_from_theta(theta):
        s_juv = 0.2 + 0.75 / (1.0 + math.exp(-theta[0]))
        g_juv = 0.2 + 0.75 / (1.0 + math.exp(-theta[1]))
        s_ad = 0.2 + 0.75 / (1.0 + math.exp(-theta[2]))
        fec = math.exp(theta[3])
        L = np.zeros((n_st, n_st), dtype=float)
        for i in range(n_st - 1):
            L[i, i] = s_juv * (1.0 - g_juv)
            L[i + 1, i] = s_juv * g_juv
        L[n_st - 1, n_st - 1] = s_ad
        L[0, max(1, n_st // 2):] = fec
        return L

    def obj(theta):
        L = _build_from_theta(theta)
        sim = np.zeros_like(mat)
        sim[:, 0] = mat[:, 0]
        for t in range(n_yr - 1):
            sim[:, t + 1] = L @ sim[:, t]
        err_mat = float(np.sum(((mat - sim) / scale) ** 2))
        err_tot = float(np.sum(((obs_tot - sim.sum(axis=0)) / scale) ** 2))
        return err_mat + 0.5 * err_tot

    res = minimize(obj, [0.5, 0.5, 1.0, -0.5], method="Nelder-Mead", options={"maxiter": 600})
    L_hat = _build_from_theta(res.x)
    sim = np.zeros_like(mat)
    sim[:, 0] = mat[:, 0]
    for t in range(n_yr - 1):
        sim[:, t + 1] = L_hat @ sim[:, t]
    pred_tot = sim.sum(axis=0)
    rss = float(np.sum((obs_tot - pred_tot) ** 2))
    ss_tot = float(np.sum((obs_tot - np.mean(obs_tot)) ** 2)) + 1e-12
    rmse = math.sqrt(rss / n_yr)
    r2 = 1.0 - rss / ss_tot
    aic = n_yr * math.log(max(1e-12, rss / n_yr)) + 8.0
    summ = summarize_leslie_model(L_hat)
    summ.update({
        "L": L_hat,
        "R2": float(r2),
        "rmse": float(rmse),
        "aic": float(aic),
        "predicted": pred_tot,
        "model": "leslie",
    })
    return summ


def compare_macro_models(
    observed: np.ndarray,
    catch: Optional[np.ndarray] = None,
    stage_matrix: Optional[np.ndarray] = None,
    params=None,
) -> dict:
    """Fit and AIC-compare Schaefer, Delay-Difference, and Generalized Leslie models."""
    obs = np.asarray(observed, dtype=float)
    c = np.zeros_like(obs) if catch is None else np.asarray(catch, dtype=float)
    sch = fit_schaefer(obs, c, dynamic=True)
    dd = fit_delay_difference(obs, c)
    les = fit_generalized_leslie_model(stage_matrix, params) if stage_matrix is not None else None

    candidates = {"schaefer": sch, "delay_difference": dd}
    if les is not None:
        candidates["leslie"] = les
    best_name = min(candidates.keys(), key=lambda k: candidates[k].get("aic", float("inf")))
    return {"schaefer": sch, "delay_difference": dd, "leslie": les, "best_model": best_name}


def retrospective_analysis(
    observed: np.ndarray, catch: Optional[np.ndarray] = None, n_peels: int = 5
) -> dict:
    """Perform retrospective peels of the Schaefer model and compute Mohn's rho."""
    obs = np.asarray(observed, dtype=float)
    c = np.zeros_like(obs) if catch is None else np.asarray(catch, dtype=float)
    full = fit_schaefer(obs, c, dynamic=True)
    full_pred = full.get("predicted", obs)
    peel_vals: List[float] = []
    full_vals: List[float] = []
    for p in range(1, n_peels + 1):
        if len(obs) - p < 4:
            break
        sub_fit = fit_schaefer(obs[:-p], c[:-p], dynamic=True)
        peel_vals.append(float(sub_fit["predicted"][-1]))
        full_vals.append(float(full_pred[len(obs) - p - 1]))
    rho = mohn_rho(peel_vals, full_vals) if peel_vals else 0.0
    return {"mohn_rho": rho, "peel_endpoints": peel_vals}


# ── Population criteria / evaluation ─────────────────────────────────


def evaluate_population_criteria(model_df, params, agent_df=None) -> dict:
    """Evaluate multi-attribute population criteria and composite score."""
    crit = params.criteria
    results: Dict[str, Any] = {}
    score = 0.0

    if len(model_df) > 0 and "n_agents" in model_df.columns:
        min_pop = float(model_df["n_agents"].min())
        max_pop = float(model_df["n_agents"].max())
        k = max(1.0, params.environment.benthic_carrying_capacity)
        pers_ok = min_pop >= crit.min_population
        no_exp = max_pop <= k * crit.max_population_multiple
        results["persistence"] = bool(pers_ok)
        results["min_population"] = int(min_pop)
        results["no_explosion"] = bool(no_exp)
        results["max_population"] = int(max_pop)
        results["peak_population"] = int(max_pop)
        if not pers_ok:
            score += crit.extinction_penalty
        if not no_exp:
            score += crit.explosion_penalty

    if len(model_df) > crit.window_steps and "mean_cw" in model_df.columns:
        recent = model_df["mean_cw"].tail(crit.window_steps)
        drift = float((recent.iloc[-1] - recent.iloc[0]) / max(1.0, recent.iloc[0]))
        results["size_drift_ok"] = abs(drift) <= crit.max_size_drift
        results["size_drift"] = drift
        score += crit.size_drift_weight * abs(drift)

    mat_rmse = 0.0
    growth_rmse = 0.0
    if agent_df is not None and len(agent_df) > 0 and "carapace_width" in agent_df.columns:
        from .species import growth_schedule, maturity_probability
        from .types import Sex

        cws = agent_df["carapace_width"].to_numpy(dtype=float)
        mats = agent_df["is_mature"].to_numpy(dtype=float) if "is_mature" in agent_df.columns else np.zeros_like(cws)
        m_mask = agent_df["sex"].astype(str).str.lower().to_numpy() == "male" if "sex" in agent_df.columns else np.ones(len(cws), dtype=bool)
        if np.any(m_mask):
            emp = compute_empirical_ogive(cws[m_mask], mats[m_mask])
            if len(emp["sizes"]) > 0:
                pred_m = np.array([maturity_probability(s, params, temp=4.0, sex=Sex.MALE) for s in emp["sizes"]])
                mat_rmse = float(np.sqrt(np.mean((emp["proportions"] - pred_m) ** 2)))
        sched = np.array(growth_schedule(Sex.MALE, params, n_stages=12), dtype=float)
        if "molt_stage" in agent_df.columns:
            obs_means = []
            pred_means = []
            for idx, st in enumerate(range(4, 16)):
                sub = cws[(agent_df["molt_stage"].to_numpy(dtype=int) == st) & m_mask]
                if len(sub) > 0 and idx < len(sched):
                    obs_means.append(float(np.mean(sub)))
                    pred_means.append(float(sched[idx]))
            if obs_means:
                growth_rmse = float(np.sqrt(np.mean((np.array(obs_means) - np.array(pred_means)) ** 2)))

    results["maturity_rmse"] = mat_rmse
    results["growth_rmse"] = growth_rmse
    score += crit.maturity_rmse_weight * mat_rmse + crit.growth_rmse_weight * growth_rmse
    results["composite_score"] = float(score)
    return results


# ── Surplus production helpers ────────────────────────────────────────


def schaefer_surplus_production(B: float, r: float, K: float) -> float:
    return r * B * (1.0 - B / K)


def mohn_rho(peel_estimates, full_estimates=None) -> float:
    if full_estimates is None:
        arr = np.asarray(peel_estimates, dtype=float)
        if len(arr) < 2:
            return 0.0
        ref = float(arr[-1])
        return float(np.mean((arr[:-1] - ref) / max(1e-9, ref)))
    if not peel_estimates or len(peel_estimates) != len(full_estimates):
        return 0.0
    diffs = []
    for p, f in zip(peel_estimates, full_estimates):
        if f != 0:
            diffs.append((p - f) / f)
    return float(np.mean(diffs)) if diffs else 0.0


# ── TethysBio Parity Extensions: IO, KDE, Ogives, Empirical & Macro ──

OBJECTIVE_PENALTY = 1e12

ANNUAL_PRODUCTION_SCHEMA = [
    "year",
    "n_agents",
    "biomass_kg",
    "mature_females",
    "mature_males",
    "legal_males",
    "mean_cw",
    "catch_total",
    "catch_biomass_kg",
    "recruits",
]


def logistic_unit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(x)))))


def logit(p: float) -> float:
    p_c = max(1e-9, min(1.0 - 1e-9, float(p)))
    return math.log(p_c / (1.0 - p_c))


def safe_objective(fn, x) -> float:
    try:
        val = float(fn(x))
        return val if math.isfinite(val) else OBJECTIVE_PENALTY
    except Exception:
        return OBJECTIVE_PENALTY


def optimize_model(fn, x0, bounds=None) -> dict:
    from scipy.optimize import minimize

    res = minimize(lambda x: safe_objective(fn, x), np.asarray(x0, dtype=float), method="Nelder-Mead")
    return {"x": res.x, "fun": float(res.fun), "success": bool(res.success)}


def optimize_blocks(fn, blocks: List[Tuple[np.ndarray, Any]], rounds: int = 2) -> dict:
    theta = np.concatenate([np.asarray(b[0], dtype=float) for b in blocks]) if blocks else np.array([])
    best = optimize_model(fn, theta)
    return best


def empty_annual_production():
    import pandas as pd

    return pd.DataFrame(columns=ANNUAL_PRODUCTION_SCHEMA)


def extract_annual_production(model_df, days_per_year: float = 365.0):
    import pandas as pd

    if model_df is None or len(model_df) == 0:
        return empty_annual_production()
    df = model_df.copy()
    if "day" in df.columns:
        df["year"] = (df["day"] // days_per_year).astype(int) + 1
    else:
        df["year"] = (np.arange(len(df)) // max(1, int(round(days_per_year)))) + 1
    grouped = df.groupby("year", as_index=False).agg(
        n_agents=("n_agents", "mean") if "n_agents" in df.columns else ("year", "count"),
        biomass_kg=("total_biomass_kg", "mean") if "total_biomass_kg" in df.columns else ("n_agents", "mean"),
        mature_females=("n_mature_females", "mean") if "n_mature_females" in df.columns else ("year", "count"),
        mature_males=("n_mature_males", "mean") if "n_mature_males" in df.columns else ("year", "count"),
        legal_males=("n_legal_males", "mean") if "n_legal_males" in df.columns else ("year", "count"),
        mean_cw=("mean_cw", "mean") if "mean_cw" in df.columns else ("year", "count"),
        catch_total=("catch_total", "max") if "catch_total" in df.columns else ("year", "count"),
        catch_biomass_kg=("catch_biomass_kg", "max") if "catch_biomass_kg" in df.columns else ("year", "count"),
        recruits=("n_juveniles", "mean") if "n_juveniles" in df.columns else ("year", "count"),
    )
    return grouped


def annual_production_targets(annual_df) -> dict:
    if annual_df is None or len(annual_df) == 0:
        return {"biomass": np.array([]), "catch": np.array([]), "recruits": np.array([])}
    return {
        "biomass": annual_df["biomass_kg"].to_numpy(dtype=float),
        "catch": annual_df["catch_biomass_kg"].to_numpy(dtype=float),
        "recruits": annual_df["recruits"].to_numpy(dtype=float),
    }


def resolve_simulation_results_path(output_dir: str = "outputs", filename: str = "annual_production.parquet") -> str:
    from pathlib import Path

    p = Path(output_dir) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def save_simulation_results(model_df, agent_df=None, output_dir: str = "outputs", days_per_year: float = 365.0) -> dict:
    from pathlib import Path
    import pandas as pd

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    annual = extract_annual_production(model_df, days_per_year=days_per_year)
    annual_path = out / "annual_production.parquet"
    model_path = out / "model_results.parquet"
    annual.to_parquet(annual_path, index=False)
    model_df.to_parquet(model_path, index=False)
    meta = pd.DataFrame([{"rows_model": len(model_df), "rows_annual": len(annual)}])
    meta_path = out / "run_metadata.parquet"
    meta.to_parquet(meta_path, index=False)
    paths = {"annual_production": str(annual_path), "model_results": str(model_path), "run_metadata": str(meta_path)}
    if agent_df is not None and len(agent_df) > 0:
        agent_path = out / "agent_results.parquet"
        agent_df.to_parquet(agent_path, index=False)
        paths["agent_results"] = str(agent_path)
    return paths


def load_simulation_results(output_dir: str = "outputs") -> dict:
    from pathlib import Path
    import pandas as pd

    out = Path(output_dir)
    res = {}
    for name in ("annual_production", "model_results", "agent_results", "run_metadata"):
        p = out / f"{name}.parquet"
        if p.exists():
            res[name] = pd.read_parquet(p)
    return res


def extract_stock_recruitment(annual_df, lag: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    if annual_df is None or len(annual_df) <= lag:
        return np.array([]), np.array([])
    sp = annual_df["mature_females"].to_numpy(dtype=float)[:-lag]
    rec = annual_df["recruits"].to_numpy(dtype=float)[lag:]
    return sp, rec


def summarize_stock_recruitment(sr_fit: dict) -> str:
    best = sr_fit.get("best_model", "ricker")
    r_fit = sr_fit.get(best, {})
    return f"Stock-Recruitment best={best} (R2={r_fit.get('r2', 0.0):.3f}, AIC={r_fit.get('aic', 0.0):.1f})"


def compute_empirical_ogive(sizes, is_mature, bin_width: float = 5.0) -> dict:
    s = np.asarray(sizes, dtype=float)
    m = np.asarray(is_mature, dtype=float)
    if len(s) == 0:
        return {"sizes": np.array([]), "proportions": np.array([]), "counts": np.array([])}
    lo = math.floor(float(np.min(s)) / bin_width) * bin_width
    hi = math.ceil(float(np.max(s)) / bin_width) * bin_width + bin_width
    edges = np.arange(lo, hi, bin_width)
    mids, props, cnts = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (s >= a) & (s < b)
        if np.any(mask):
            mids.append(0.5 * (a + b))
            props.append(float(np.mean(m[mask])))
            cnts.append(int(np.sum(mask)))
    return {"sizes": np.array(mids), "proportions": np.array(props), "counts": np.array(cnts)}


def cumulative_maturity_curves(sizes, params) -> dict:
    from .species import maturity_probability
    from .types import Sex

    grid = np.asarray(sizes, dtype=float)
    return {
        "sizes": grid,
        "male": np.array([maturity_probability(x, Sex.MALE, params) for x in grid]),
        "female": np.array([maturity_probability(x, Sex.FEMALE, params) for x in grid]),
    }


def maturity_ogive_residuals(sizes, is_mature, sex, params) -> np.ndarray:
    from .species import maturity_probability

    emp = compute_empirical_ogive(sizes, is_mature)
    if len(emp["sizes"]) == 0:
        return np.array([])
    pred = np.array([maturity_probability(x, sex, params) for x in emp["sizes"]])
    return emp["proportions"] - pred


def maturity_ogive_mismatch(sizes, is_mature, sex, params) -> float:
    res = maturity_ogive_residuals(sizes, is_mature, sex, params)
    return float(np.sqrt(np.mean(res ** 2))) if len(res) > 0 else 0.0


def stage_at_maturity_fraction(params, sex=None) -> float:
    from .species import maturity_probability
    from .types import Sex

    s_use = sex if sex is not None else Sex.MALE
    l50 = params.species.cw50_maturity if s_use == Sex.MALE else params.species.cw50_maturity_female
    return float(l50)


def summarize_population(agents: list, params) -> dict:
    if not agents:
        return {"n": 0, "mean_cw": 0.0, "biomass_kg": 0.0, "maturity_rate": 0.0}
    cws = [a.organism.carapace_width for a in agents]
    masses = [a.organism.mass for a in agents]
    mats = [1.0 if a.organism.is_mature else 0.0 for a in agents]
    return {
        "n": len(agents),
        "mean_cw": float(np.mean(cws)),
        "biomass_kg": float(np.sum(masses) / 1000.0),
        "maturity_rate": float(np.mean(mats)),
    }


def gaussian_kde(x, grid=None, bandwidth: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=float)
    if grid is None:
        g = np.linspace(5.0, 160.0, 120)
    else:
        g = np.asarray(grid, dtype=float)
    if len(arr) == 0:
        return g, np.zeros_like(g)
    bw = max(0.5, float(bandwidth))
    diffs = (g[:, None] - arr[None, :]) / bw
    dens = np.mean(np.exp(-0.5 * diffs ** 2), axis=1) / (bw * math.sqrt(2.0 * math.pi))
    return g, dens


def derive_stage_proportions_kde(sizes, params, sex=None) -> np.ndarray:
    from .types import Sex

    s_use = sex if sex is not None else Sex.MALE
    grid, dens = gaussian_kde(sizes)
    bounds = params.species.stage_size_range_lower_male if s_use == Sex.MALE else params.species.stage_size_range_lower_female
    uppers = params.species.stage_size_range_upper_male if s_use == Sex.MALE else params.species.stage_size_range_upper_female
    props = []
    for lo, hi in zip(bounds, uppers):
        mask = (grid >= lo) & (grid < hi)
        props.append(float(np.trapezoid(dens[mask], grid[mask])) if np.any(mask) else 0.0)
    arr = np.asarray(props, dtype=float)
    tot = float(np.sum(arr))
    return arr / tot if tot > 0 else arr


def generate_mock_survey_data(n: int, params, seed: int = 42) -> dict:
    import random
    from .species import sample_population

    rng = random.Random(seed)
    orgs = sample_population(n, params, rng=rng)
    return {
        "carapace_width": np.array([o.carapace_width for o in orgs]),
        "is_mature": np.array([o.is_mature for o in orgs]),
        "sex": np.array([o.sex.value for o in orgs]),
        "molt_stage": np.array([o.molt_stage for o in orgs]),
    }


def fit_empirical_biology_model(survey_data: dict, params) -> dict:
    cws = np.asarray(survey_data.get("carapace_width", []), dtype=float)
    mats = np.asarray(survey_data.get("is_mature", []), dtype=float)
    sexes = np.asarray(survey_data.get("sex", []), dtype=str)
    m_mask = sexes == "male"
    f_mask = sexes == "female"
    l50_m = float(np.median(cws[m_mask & (mats > 0.5)])) if np.any(m_mask & (mats > 0.5)) else params.species.cw50_maturity
    l50_f = float(np.median(cws[f_mask & (mats > 0.5)])) if np.any(f_mask & (mats > 0.5)) else params.species.cw50_maturity_female
    return {"l50_male": l50_m, "l50_female": l50_f, "n_samples": int(len(cws))}


def estimate_demographics(survey_data: dict) -> dict:
    cws = np.asarray(survey_data.get("carapace_width", []), dtype=float)
    mats = np.asarray(survey_data.get("is_mature", []), dtype=float)
    return {
        "mean_cw": float(np.mean(cws)) if len(cws) else 0.0,
        "maturity_fraction": float(np.mean(mats)) if len(mats) else 0.0,
        "count": int(len(cws)),
    }


def sample_organism_traits(n: int, params, seed: int = 42) -> list:
    import random

    rng = random.Random(seed)
    qmin = params.organism.trait_quantile_min
    qmax = params.organism.trait_quantile_max
    return [
        {
            "growth": rng.uniform(qmin, qmax),
            "maturity": rng.uniform(qmin, qmax),
            "fecundity": rng.uniform(qmin, qmax),
            "mortality": rng.uniform(qmin, qmax),
        }
        for _ in range(n)
    ]


def summarize_empirical_model(model: dict) -> str:
    return f"EmpiricalBiology(L50_M={model.get('l50_male', 0):.1f}mm, L50_F={model.get('l50_female', 0):.1f}mm)"


def fit_population_dynamics(observed, catch=None) -> dict:
    return compare_macro_models(observed, catch=catch)


def summarize_population_dynamics(fit: dict) -> str:
    best = fit.get("best_model", "schaefer")
    return f"PopulationDynamics(best={best}, R2={fit.get(best, {}).get('r2', 0.0):.3f})"


def summarize_macro_comparison(comp: dict) -> str:
    return summarize_population_dynamics(comp)


def extract_annual_stage_matrix(agent_df, stages: list) -> np.ndarray:
    if agent_df is None or len(agent_df) == 0 or "molt_stage" not in agent_df.columns:
        return np.zeros((len(stages), 1), dtype=float)
    steps = sorted(agent_df.index.get_level_values(0).unique())
    mat = np.zeros((len(stages), len(steps)), dtype=float)
    for j, st in enumerate(steps):
        sub = agent_df.xs(st, level=0)
        for i, s_idx in enumerate(stages):
            mat[i, j] = float((sub["molt_stage"] == s_idx).sum())
    return mat


def summarize_retrospective(retro: dict) -> str:
    return f"Retrospective(Mohn's rho={retro.get('mohn_rho', 0.0):.4f}, peels={len(retro.get('peel_endpoints', []))})"
