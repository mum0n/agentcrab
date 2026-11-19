"""Species behavior and allometric functions for AgentCrab.

Snow crab (Chionoecetes opilio) biological rules aligned with TethysBio:
- Sexual dimorphism: males grow larger (max 160 mm) than females (max 95 mm)
- Maturity ogives differ by sex: male L50 = 91.8 mm, female L50 = 51.6 mm
- Fecundity: base-10 Moncton primiparous/multiparous or St. Marie power-law
- Growth: sex-specific Hiatt logarithmic growth with terminal maturation penalty
- Terminal morphometric maturity is irreversible
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Protocol, Union

from .types import IBMParameters, LifeStage, Organism, Sex


def _norm_ppf(p: float) -> float:
    """Acklam rational approximation to the standard normal inverse CDF."""
    p = max(1e-9, min(1.0 - 1e-9, float(p)))
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def _stage_to_int(stage: Union[int, LifeStage]) -> int:
    if isinstance(stage, int):
        return stage
    mapping = {
        LifeStage.EGG: 0,
        LifeStage.ZOEA: 1,
        LifeStage.MEGALOPA: 3,
        LifeStage.JUVENILE: 8,
        LifeStage.ADULT: 13,
    }
    return mapping.get(stage, 12)


@dataclass
class StageDefinition:
    """Stage definition for generalized Leslie matrix modeling."""
    stage_index: int
    label: str
    sex: str
    lower_mm: float
    upper_mm: float
    mean_mm: float


class SpeciesBehavior(Protocol):
    """Protocol for species-specific biological behavior."""

    def stage_size_lower_bound(
        self, stage: Union[int, LifeStage], params: IBMParameters, sex: Sex = Sex.MALE
    ) -> float: ...
    def growth_increment(
        self, organism: Organism, params: IBMParameters, temp: float
    ) -> float: ...
    def maturity_probability(
        self, organism: Organism, params: IBMParameters, temp: float
    ) -> float: ...
    def fecundity(self, organism: Organism, params: IBMParameters) -> float: ...
    def max_size(self, sex: Sex, params: IBMParameters) -> float: ...


@dataclass
class SpeciesBehaviorImpl:
    """Default implementation of SpeciesBehavior for snow crab matching TethysBio."""

    def max_size(self, sex: Sex, params: IBMParameters) -> float:
        sp = params.species
        if sex == Sex.FEMALE:
            return sp.max_size_female
        return sp.max_size_male

    def stage_size_lower_bound(
        self, stage: Union[int, LifeStage], params: IBMParameters, sex: Sex = Sex.MALE
    ) -> float:
        sp = params.species
        s_idx = _stage_to_int(stage)
        if s_idx <= 0:
            return params.organism.egg_size_mm

        # Pre-benthic larval lookup before benthic_anchor_stage
        if s_idx < sp.benthic_anchor_stage:
            bps = sp.size_lb_immature_breakpoints
            szs = sp.size_lb_immature_sizes
            for i, bp in enumerate(bps):
                if s_idx < bp:
                    return szs[i]
            return szs[-1]

        if sex == Sex.FEMALE:
            linf = sp.max_size_female
            intercept = sp.size_lb_female_intercept
            decay = sp.size_lb_female_decay
            anchor = sp.size_lb_female_anchor_stage
        else:
            linf = sp.max_size_male
            intercept = sp.size_lb_male_intercept
            decay = sp.size_lb_male_decay
            anchor = sp.size_lb_male_anchor_stage

        ln_linf = math.log(linf)
        return math.exp(ln_linf + (intercept - ln_linf) * math.exp(-decay * (s_idx - anchor)))

    def growth_increment(
        self,
        organism: Organism,
        params: IBMParameters,
        temp: float,
        is_maturing: bool = False,
        condition: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> float:
        cw = max(1.0, organism.carapace_width)
        next_cw = predict_postgrowth_size(
            cw,
            params,
            temp,
            organism.sex,
            is_maturing=is_maturing,
            condition=condition,
            growth_quantile=organism.trait_growth_quantile,
            rng=rng,
        )
        return next_cw - cw

    def maturity_probability(
        self, organism: Organism, params: IBMParameters, temp: float
    ) -> float:
        sp = params.species
        cw = organism.carapace_width
        if cw < sp.maturity_size_gate:
            return 0.0

        z_q = _norm_ppf(organism.trait_maturity_quantile)
        if organism.sex == Sex.FEMALE:
            l50_base = sp.cw50_maturity_female + z_q * sp.maturity_l50_female_sd
            slope_base = max(0.5, sp.maturity_slope_female + z_q * sp.maturity_slope_female_sd)
        else:
            l50_base = sp.cw50_maturity + z_q * sp.maturity_l50_male_sd
            slope_base = max(0.5, sp.maturity_slope_male + z_q * sp.maturity_slope_male_sd)

        l50_shifted = max(
            sp.maturity_size_gate,
            l50_base + sp.maturity_l50_temp_slope * (temp - sp.maturity_temp_reference),
        )
        logit_val = slope_base * (math.log(max(1.0, cw)) - math.log(l50_shifted))
        logit_val = max(-35.0, min(35.0, logit_val))
        return 1.0 / (1.0 + math.exp(-logit_val))

    def fecundity(
        self,
        organism: Organism,
        params: IBMParameters,
        rng: Optional[random.Random] = None,
    ) -> float:
        sp = params.species
        if organism.sex != Sex.FEMALE or not organism.is_mature:
            return 0.0

        cw = max(1.0, organism.carapace_width)
        shell_age_years = organism.shell_age_days / params.simulation.days_per_year
        r = rng if rng is not None else random

        if sp.fecundity_parameterization.lower() in ("st.marie", "stmarie"):
            # Base-10 log-linear St. Marie form
            z_q = _norm_ppf(organism.trait_fecundity_quantile)
            log10_eggs = (
                sp.fecundity_stmarie_coefficient
                + z_q * sp.fecundity_intercept_sd
                + sp.fecundity_stmarie_exponent * math.log10(cw)
            )
            eggs = 10.0 ** log10_eggs
        else:
            # Moncton base-10 primiparous vs multiparous
            is_primi = shell_age_years <= sp.primiparous_shell_age_max
            if is_primi:
                intercept = sp.fecundity_primiparous_intercept
                slope = sp.fecundity_primiparous_slope
                err_sd = sp.fecundity_primiparous_err
            else:
                intercept = sp.fecundity_multiparous_intercept
                slope = sp.fecundity_multiparous_slope
                err_sd = sp.fecundity_multiparous_err
            z_q = _norm_ppf(organism.trait_fecundity_quantile)
            noise = r.gauss(0.0, err_sd) if rng is not None else 0.0
            log10_eggs = (
                intercept
                + z_q * sp.fecundity_intercept_sd
                + slope * math.log10(cw)
                + noise
            )
            eggs = 10.0 ** log10_eggs

        # Senescence discount for older females
        if shell_age_years > sp.fecundity_senescence_threshold:
            discount = max(
                sp.fecundity_senescence_floor,
                1.0 - sp.fecundity_senescence_rate * (shell_age_years - sp.fecundity_senescence_threshold),
            )
            eggs *= discount

        return max(float(params.phenology.minimum_clutch_size), eggs)


def default_species_behavior() -> SpeciesBehaviorImpl:
    return SpeciesBehaviorImpl()


# ── Allometric helpers ────────────────────────────────────────────────


def max_size(sex: Sex, params: IBMParameters) -> float:
    return default_species_behavior().max_size(sex, params)


def calculate_mass(
    cw: float,
    sex: Sex,
    is_mature: bool,
    params: IBMParameters,
    include_eggs: bool = False,
    egg_count: float = 0.0,
    rng: Optional[random.Random] = None,
) -> float:
    """Compute wet mass (g) from carapace width using sex- and maturity-specific allometry."""
    sp = params.species
    if sex == Sex.FEMALE:
        alpha = sp.mass_alpha_female_mature if is_mature else sp.mass_alpha_female_immature
        beta = sp.mass_beta_female_mature if is_mature else sp.mass_beta_female_immature
    else:
        alpha = sp.mass_alpha_male_mature if is_mature else sp.mass_alpha_male_immature
        beta = sp.mass_beta_male_mature if is_mature else sp.mass_beta_male_immature

    mass = alpha * (max(0.1, cw) ** beta)
    if rng is not None and sp.mass_sigma > 0.0:
        mass *= math.exp(rng.gauss(0.0, sp.mass_sigma))
    if include_eggs and sex == Sex.FEMALE and egg_count > 0.0:
        mass += egg_count * params.organism.egg_mass_g
    return mass


def llbm(stage_or_cw: Union[int, float], params: IBMParameters) -> float:
    """Lower size bound (mm) for male snow crab at stage."""
    return default_species_behavior().stage_size_lower_bound(
        int(round(stage_or_cw)), params, Sex.MALE
    )


def llbf(stage_or_cw: Union[int, float], params: IBMParameters) -> float:
    """Lower size bound (mm) for female snow crab at stage."""
    return default_species_behavior().stage_size_lower_bound(
        int(round(stage_or_cw)), params, Sex.FEMALE
    )


def get_stage_size(
    stage: Union[int, LifeStage],
    params: IBMParameters,
    sex: Sex = Sex.MALE,
    mode: str = "mid",
    rng: Optional[random.Random] = None,
) -> float:
    """Return lower bound, upper bound, midpoint, or Beta-sampled size (mm) for a stage."""
    beh = default_species_behavior()
    s_idx = _stage_to_int(stage)
    lb = beh.stage_size_lower_bound(s_idx, params, sex)
    ub = beh.stage_size_lower_bound(s_idx + 1, params, sex)
    ub = max(lb * 1.05, min(ub, beh.max_size(sex, params)))

    if mode == "lb":
        return lb
    if mode == "ub":
        return ub
    if mode == "random":
        r = rng if rng is not None else random
        u = r.betavariate(params.organism.stage_size_sample_a, params.organism.stage_size_sample_b)
        return lb + u * (ub - lb)
    return 0.5 * (lb + ub)


def maturity_probability(organism, params: IBMParameters, temp: float = 4.0, sex: Sex = Sex.MALE) -> float:
    if isinstance(organism, (int, float)):
        tmp = Organism(
            unique_id=0,
            sex=sex,
            stage=LifeStage.JUVENILE,
            carapace_width=float(organism),
            molt_stage=10,
        )
        return default_species_behavior().maturity_probability(tmp, params, temp)
    return default_species_behavior().maturity_probability(organism, params, temp)


def predict_postgrowth_size(
    cw: float,
    params: IBMParameters,
    temp: float,
    sex: Sex = Sex.MALE,
    is_maturing: bool = False,
    condition: float = 1.0,
    growth_quantile: float = 0.5,
    rng: Optional[random.Random] = None,
) -> float:
    """Predict carapace width (mm) after ecdysis using TethysBio Hiatt log-growth."""
    sp = params.species
    z1 = _norm_ppf(growth_quantile)
    z2 = rng.gauss(0.0, 1.0) if rng is not None else 0.0

    if sex == Sex.FEMALE:
        b_int = sp.growth_int_female + z1 * sp.growth_beta1_female
        b_slp = sp.growth_slp_female + z1 * sp.growth_beta2_female + z2 * sp.growth_beta3_female
        pen_mean = sp.growth_maturation_penalty_female_mean
        pen_sd = sp.growth_maturation_penalty_female_sd
        max_sz = sp.max_size_female
    else:
        b_int = sp.growth_int_male + z1 * sp.growth_beta1_male
        b_slp = sp.growth_slp_male + z1 * sp.growth_beta2_male + z2 * sp.growth_beta3_male
        pen_mean = sp.growth_maturation_penalty_male_mean
        pen_sd = sp.growth_maturation_penalty_male_sd
        max_sz = sp.max_size_male

    ln_cw = math.log(max(1.0, cw))
    mat_pen = 0.0
    if is_maturing:
        mat_pen = pen_mean + (rng.gauss(0.0, pen_sd) if rng is not None else 0.0)

    resid = rng.gauss(0.0, sp.growth_residual_sd) if rng is not None else 0.0
    ln_next = (
        b_int
        + b_slp * ln_cw
        + mat_pen
        + sp.growth_condition_coefficient * (condition - 1.0)
        + sp.growth_temperature_coefficient * (temp - sp.growth_temperature_reference)
        + resid
    )

    min_exp = sp.growth_min_expansion_mature if is_maturing else sp.growth_min_expansion_immature
    ln_next = max(ln_next, ln_cw + math.log(1.0 + min_exp))
    next_cw = math.exp(ln_next)
    return min(next_cw, max_sz)


def build_stages(params: IBMParameters) -> List[StageDefinition]:
    """Build StageDefinition list for generalized Leslie matrix modeling."""
    sp = params.species
    stages: List[StageDefinition] = []
    for i, s in enumerate(sp.stage_size_range_stage1):
        stages.append(
            StageDefinition(
                stage_index=int(round(s)),
                label=f"Stage {int(round(s))}",
                sex="pooled",
                lower_mm=sp.stage_size_range_lower_stage1[i],
                upper_mm=sp.stage_size_range_upper_stage1[i],
                mean_mm=sp.stage_size_range_mean_stage1[i],
            )
        )
    for i, s in enumerate(sp.stage_size_range_stage_male):
        stages.append(
            StageDefinition(
                stage_index=int(round(s)),
                label=f"M{int(round(s))}",
                sex="male",
                lower_mm=sp.stage_size_range_lower_male[i],
                upper_mm=sp.stage_size_range_upper_male[i],
                mean_mm=sp.stage_size_range_mean_male[i],
            )
        )
    for i, s in enumerate(sp.stage_size_range_stage_female):
        stages.append(
            StageDefinition(
                stage_index=int(round(s)),
                label=f"F{int(round(s))}",
                sex="female",
                lower_mm=sp.stage_size_range_lower_female[i],
                upper_mm=sp.stage_size_range_upper_female[i],
                mean_mm=sp.stage_size_range_mean_female[i],
            )
        )
    return stages


def larval_stage_count() -> int:
    """Return number of pelagic larval stages (z1, z2, m)."""
    return 3


def selectivity(cw: float, params: IBMParameters) -> float:
    """Logistic gear selectivity curve matching TethysBio::selectivity."""
    s50 = params.species.selectivity_s50
    sigma = max(1e-6, params.species.selectivity_s50_sigma)
    return 1.0 / (1.0 + math.exp(-(cw - s50) / sigma))


def growth_schedule(
    sex: Sex,
    params: IBMParameters,
    temp: float = 4.0,
    condition: float = 1.0,
    n_stages: int = 16,
) -> List[float]:
    """Compute deterministic mean carapace width trajectory across benthic instars."""
    cw = get_stage_size(4, params, sex, mode="mid")
    sizes = [cw]
    for _ in range(1, n_stages):
        cw = predict_postgrowth_size(cw, params, temp, sex=sex, is_maturing=False, condition=condition, rng=None)
        sizes.append(cw)
    return sizes


def growth_simulate(
    sex: Sex,
    params: IBMParameters,
    temp: float = 4.0,
    condition: float = 1.0,
    n_stages: int = 16,
    rng: Optional[random.Random] = None,
) -> List[float]:
    """Simulate stochastic individual carapace width trajectory across benthic instars."""
    r = rng if rng is not None else random
    cw = get_stage_size(4, params, sex, mode="random", rng=r)
    sizes = [cw]
    for _ in range(5, 4 + n_stages):
        p_mat = maturity_probability(cw, params, temp=temp, sex=sex)
        is_mat = r.random() < p_mat
        cw = predict_postgrowth_size(cw, params, temp, sex=sex, is_maturing=is_mat, condition=condition, rng=r)
        sizes.append(cw)
        if is_mat:
            break
    return sizes


def sample_population(
    n: int,
    params: IBMParameters,
    rng: Optional[random.Random] = None,
) -> List[Organism]:
    """Sample a synthetic population of benthic Organism agents across stages."""
    r = rng if rng is not None else random
    out: List[Organism] = []
    for i in range(n):
        sex = Sex.FEMALE if r.random() < 0.5 else Sex.MALE
        max_s = params.species.max_stage_female if sex == Sex.FEMALE else params.species.max_stage_male
        s = r.randint(4, max_s)
        cw = get_stage_size(s, params, sex, mode="random", rng=r)
        is_mat = r.random() < maturity_probability(cw, params, temp=4.0, sex=sex)
        stage = LifeStage.ADULT if is_mat or s >= params.species.juvenile_max_stage else LifeStage.JUVENILE
        out.append(
            Organism(
                unique_id=i + 1,
                sex=sex,
                stage=stage,
                carapace_width=cw,
                molt_stage=s,
                is_mature=is_mat,
                position=(r.randint(0, params.simulation.grid_nx - 1), r.randint(0, params.simulation.grid_ny - 1)),
            )
        )
    return out
