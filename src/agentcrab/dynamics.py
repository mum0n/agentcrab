"""Organism dynamics, bioenergetics, ecdysis, and competing-risks mortality for AgentCrab."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple, Union

from .types import (
    AggregateGroup,
    IBMParameters,
    LifeStage,
    Organism,
    Sex,
)
from .species import (
    _norm_ppf,
    calculate_mass,
    default_species_behavior,
    get_stage_size,
    maturity_probability,
    predict_postgrowth_size,
)


def _poisson_cdf(k: int, lam: float) -> float:
    """Compute P(X <= k) for X ~ Poisson(lam)."""
    if k < 0:
        return 0.0
    if lam <= 0.0:
        return 1.0
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(1.0, total)


def calculate_energy_change(
    organism: Organism,
    params: IBMParameters,
    temp: float,
    food_avail: float = 1.0,
) -> float:
    """Calculate daily net energy change (J/day-equivalent) matching TethysBio bioenergetics."""
    if not params.toggles.use_energetics:
        return 0.0

    org_p = params.organism
    mass = max(1e-3, organism.mass)
    t_opt = org_p.metabolic_optimal_temperature
    t_sig = max(0.1, org_p.metabolic_optimal_sigma)

    # Unimodal thermal performance curve for feeding
    f_temp = math.exp(-0.5 * ((temp - t_opt) / t_sig) ** 2)
    half_sat = max(1e-6, params.environment.half_saturation_food)
    f_avail = max(0.0, food_avail)
    holling = f_avail / (f_avail + half_sat)

    intake = org_p.intake_rate_base * (mass ** org_p.metabolic_mass_exponent) * holling * f_temp
    assim = intake * params.environment.assimilation_efficiency * (1.0 - org_p.specific_dynamic_action)

    # Kleiber maintenance metabolism
    maint = org_p.metabolic_rate_base * (mass ** org_p.metabolic_mass_exponent) * math.exp(
        org_p.metabolic_temperature_coefficient * temp
    )
    return assim - maint


def min_intermolt_days(organism_or_stage: Union[Organism, int], params: IBMParameters) -> float:
    """Stage-indexed minimum intermolt duration (days)."""
    s = organism_or_stage.molt_stage if isinstance(organism_or_stage, Organism) else int(organism_or_stage)
    larval = params.species.intermolt_days_larval
    if 0 <= s < len(larval):
        return float(larval[s])
    sched = params.species.intermolt_days_schedule
    if not sched:
        return float(params.phenology.min_intermolt_days)
    idx = max(0, min(s, len(sched) - 1))
    return float(sched[idx])


def required_degree_days(organism_or_stage: Union[Organism, int], params: IBMParameters) -> float:
    """Stage-indexed required degree-days for ecdysis."""
    s = organism_or_stage.molt_stage if isinstance(organism_or_stage, Organism) else int(organism_or_stage)
    larval = params.species.degree_days_larval
    if 0 <= s < len(larval):
        return float(larval[s])
    sched = params.species.degree_days_schedule
    if not sched:
        return float(params.phenology.required_degree_days)
    idx = max(0, min(s, len(sched) - 1))
    return float(sched[idx])


def _in_season(day_of_year: float, start: float, end: float) -> bool:
    if start <= end:
        return start <= day_of_year <= end
    return day_of_year >= start or day_of_year <= end


def molt_probability(
    organism: Organism,
    params: IBMParameters,
    temp: float = 4.0,
    day_of_year: Optional[float] = None,
    dt: float = 1.0,
) -> float:
    """Compute step probability of ecdysis with terminal-molt, seasonal, energetic, and thermal gates."""
    if not params.toggles.use_molting or organism.is_mature:
        return 0.0

    max_stage = (
        params.species.max_stage_female
        if organism.sex == Sex.FEMALE
        else params.species.max_stage_male
    )
    if organism.molt_stage >= max_stage:
        return 0.0

    min_days = min_intermolt_days(organism, params)
    if organism.shell_age_days < min_days:
        return 0.0

    req_dd = required_degree_days(organism, params)
    if params.toggles.use_degree_days and req_dd > 0.0 and organism.degree_days_accumulated < req_dd:
        return 0.0

    ph = params.phenology
    if day_of_year is not None and organism.molt_stage >= ph.molt_season_min_stage:
        if not _in_season(day_of_year, ph.molt_season_start, ph.molt_season_end):
            return 0.0

    org_p = params.organism
    energy_req = max(org_p.energy_molt_floor, org_p.energy_molt_slope * organism.carapace_width)
    if params.toggles.use_energetics and organism.energy_reserves < energy_req:
        return 0.0

    # Ecdysial thermal inhibition above molt_thermal_onset
    thermal_factor = 1.0
    if temp > ph.molt_thermal_onset:
        bw = max(0.1, ph.molt_thermal_bandwidth)
        thermal_factor = math.exp(-0.5 * ((temp - ph.molt_thermal_onset) / bw) ** 2)

    dd_progress = (organism.degree_days_accumulated / max(1.0, req_dd)) if req_dd > 0.0 else 1.0
    progress = min(org_p.molt_progress_saturation, max(1.0, dd_progress))
    lam = org_p.molt_hazard_coefficient * progress * thermal_factor
    return 1.0 - math.exp(-max(0.0, lam) * dt)


def check_molt_conditions(
    organism: Organism,
    params: IBMParameters,
    temp: float,
    day_of_year: Optional[float] = None,
    dt: float = 1.0,
    rng: Optional[random.Random] = None,
) -> bool:
    """Return True if organism undergoes ecdysis in this step."""
    p = molt_probability(organism, params, temp=temp, day_of_year=day_of_year, dt=dt)
    if p <= 0.0:
        return False
    r = rng if rng is not None else random
    return r.random() < p


def perform_molt(
    organism: Organism,
    params: IBMParameters,
    temp: float,
    current_time: float = 0.0,
    condition: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Organism:
    """Perform ecdysis-linked growth and terminal molt evaluation."""
    r = rng if rng is not None else random
    max_stage = (
        params.species.max_stage_female
        if organism.sex == Sex.FEMALE
        else params.species.max_stage_male
    )
    next_stage = min(max_stage, organism.molt_stage + 1)

    prob_mat = maturity_probability(organism, params, temp)
    is_maturing = (not organism.is_mature) and (
        (r.random() < prob_mat) or (next_stage >= max_stage)
    )

    new_cw = predict_postgrowth_size(
        organism.carapace_width,
        params,
        temp,
        organism.sex,
        is_maturing=is_maturing,
        condition=condition,
        growth_quantile=organism.trait_growth_quantile,
        rng=rng,
    )

    organism.carapace_width = new_cw
    organism.molt_stage = next_stage
    if next_stage >= params.species.juvenile_max_stage or is_maturing:
        organism.stage = LifeStage.ADULT
    if is_maturing:
        organism.is_mature = True
        if organism.sex == Sex.FEMALE:
            organism.is_mating_ready = True

    organism.shell_age_days = 0.0
    organism.degree_days_accumulated = 0.0
    organism.last_growth_event_time = current_time
    organism.is_soft_shell = True
    organism.state = "molting"
    organism.energy_reserves *= params.organism.molt_energy_retention
    organism.mass = calculate_mass(
        new_cw,
        organism.sex,
        organism.is_mature,
        params,
        include_eggs=organism.is_ovigerous,
        egg_count=organism.egg_count,
        rng=rng,
    )
    return organism


def mortality_components(
    organism: Organism,
    params: IBMParameters,
    temp: float,
    density: float,
    dt: float = 1.0,
    n_measured: Optional[float] = None,
    k_cap: Optional[float] = None,
) -> Dict[str, float]:
    """Return dt-scaled competing-risks hazard components matching TethysBio."""
    m = params.mortality
    dpy = params.simulation.days_per_year
    cw = organism.carapace_width

    # 1. Baseline stage-specific hazard * individual trait * benthic density feedback
    rates = m.baseline_rate_by_stage
    idx = max(0, min(organism.molt_stage, len(rates) - 1))
    trait_mult = math.exp(m.baseline_trait_sigma * _norm_ppf(organism.trait_mortality_quantile))
    h_base = (rates[idx] / dpy) * trait_mult * dt
    if is_empirically_measured(organism):
        k = k_cap if k_cap is not None else params.environment.benthic_carrying_capacity
        if k > 0.0:
            n_eff = n_measured if n_measured is not None else density
            h_base *= 1.0 + (max(0.0, n_eff) / k) ** m.density_feedback_gamma

    # 2. Size-attenuated predation
    h_pred = (
        (m.rate_predation / dpy)
        * math.exp(-m.predation_size_decay * max(0.0, cw - m.predation_size_reference))
        * dt
    )

    # 3. Post-molt soft-shell vulnerability
    h_soft = 0.0
    if organism.is_soft_shell and organism.shell_age_days <= m.soft_shell_duration:
        h_soft = (
            (m.rate_soft_shell / dpy)
            * math.exp(-organism.shell_age_days / max(1.0, m.soft_shell_decay_days))
            * dt
        )

    # 4. Starvation (energy deficit below critical pool)
    h_starve = 0.0
    if params.toggles.use_energetics and organism.energy_reserves < 0.0:
        crit = max(1e-3, m.starvation_critical_fraction * params.organism.initial_energy)
        deficit = -organism.energy_reserves
        h_starve = (m.rate_starvation / dpy) * (1.0 - math.exp(-deficit / crit)) * dt

    # 5. Thermal stress above temp_thermal_stress
    h_thermal = 0.0
    if params.toggles.use_thermal_stress and temp > m.temp_thermal_stress:
        h_thermal = (m.rate_thermal / dpy) * ((temp - m.temp_thermal_stress) ** m.thermal_exponent) * dt

    # 6. Senescence (terminal-stage mature crabs past senescence_age_threshold)
    h_senescence = 0.0
    max_stage = (
        params.species.max_stage_female
        if organism.sex == Sex.FEMALE
        else params.species.max_stage_male
    )
    if params.toggles.use_senescence and organism.is_mature and organism.molt_stage >= max_stage:
        sa_y = organism.shell_age_days / dpy
        if sa_y > m.senescence_age_threshold:
            h_senescence = (
                (m.rate_senescence / dpy)
                * math.exp(m.senescence_exp_rate * (sa_y - m.senescence_age_threshold))
                * dt
            )

    # 7. Cannibalism targeting small juveniles (< cannibalism_size_threshold)
    h_cannibal = 0.0
    if params.toggles.use_cannibalism and cw < m.cannibalism_size_threshold:
        d_ratio = min(m.cannibalism_density_cap, max(0.0, density) / max(1e-6, m.cannibalism_density_reference))
        h_cannibal = (m.rate_cannibalism / dpy) * d_ratio * dt

    # 8. Post-handling / discard mortality (when recently handled within handling_mortality_days)
    h_handling = 0.0
    if getattr(organism, "last_growth_event_time", 0.0) < 0.0 and getattr(m, "rate_handling", 0.0) > 0.0:
        h_handling = (m.rate_handling / dpy) * dt

    return {
        "base": h_base,
        "predation": h_pred,
        "soft_shell": h_soft,
        "starvation": h_starve,
        "thermal": h_thermal,
        "senescence": h_senescence,
        "cannibalism": h_cannibal,
        "handling": h_handling,
    }


def calculate_mortality_probability(
    organism: Organism,
    params: IBMParameters,
    temp: float,
    density: float,
    n_measured: Optional[float] = None,
    k_cap: Optional[float] = None,
    dt: float = 1.0,
) -> float:
    """Compute competing-risks total mortality probability 1 - exp(-sum(h_i))."""
    hazards = mortality_components(
        organism, params, temp, density, dt=dt, n_measured=n_measured, k_cap=k_cap
    )
    h_total = sum(hazards.values())
    if h_total <= 0.0:
        return 0.0
    return min(params.mortality.hazard_cap, 1.0 - math.exp(-h_total))


def dominant_mortality_cause(hazards: Dict[str, float]) -> str:
    """Return the mortality cause with the largest hazard contribution."""
    return max(hazards.items(), key=lambda kv: kv[1])[0]


def shell_age_years(organism: Organism) -> float:
    return organism.shell_age_days / 365.0


def shell_age(organism: Organism, params: Optional[IBMParameters] = None) -> float:
    """Return shell age in years, offset by phenology when available."""
    dpy = params.simulation.days_per_year if params is not None else 365.0
    return organism.shell_age_days / max(1.0, dpy)


def shell_condition(organism: Organism, params: Optional[IBMParameters] = None) -> int:
    """Return integer carapace condition code 1..5 using phenology.shell_condition_days."""
    if organism.is_soft_shell:
        return 1
    thresholds = (
        params.phenology.shell_condition_days
        if params is not None and hasattr(params.phenology, "shell_condition_days")
        else [30.0, 365.0, 730.0, 1277.5]
    )
    sa = organism.shell_age_days
    for idx, bound in enumerate(thresholds, start=1):
        if sa < bound:
            return idx
    return len(thresholds) + 1


def individuation_stage(params: IBMParameters) -> int:
    """Return the molt stage index at which aggregate cohorts individuate into spatial agents."""
    return int(params.species.juvenile_max_stage)


def stage_schedule_lookup(schedule: list, stage_idx: int, default_val: float = 365.0) -> float:
    """Safely look up a 0-based stage schedule value."""
    if not schedule:
        return float(default_val)
    idx = max(0, min(int(stage_idx), len(schedule) - 1))
    return float(schedule[idx])


def merge_cohort_groups(groups: List[AggregateGroup]) -> List[AggregateGroup]:
    """Merge aggregate cohort groups sharing the same stage_idx and cell position."""
    merged: Dict[Tuple[int, Tuple[int, int]], AggregateGroup] = {}
    for g in groups:
        if g.abundance <= 0.0:
            continue
        key = (g.stage_idx, tuple(g.pos))
        if key not in merged:
            merged[key] = AggregateGroup(
                stage=g.stage,
                count=int(round(g.abundance)),
                mean_size=g.mean_size,
                age_days=g.age_days,
                degree_days_accumulated=g.degree_days_accumulated,
                id=g.id,
                stage_idx=g.stage_idx,
                abundance=float(g.abundance),
                mass=g.mass,
                last_growth_time=g.last_growth_time,
                pos=tuple(g.pos),
            )
        else:
            existing = merged[key]
            tot = existing.abundance + g.abundance
            if tot > 0.0:
                existing.mean_size = (existing.mean_size * existing.abundance + g.mean_size * g.abundance) / tot
                existing.age_days = (existing.age_days * existing.abundance + g.age_days * g.abundance) / tot
                existing.degree_days_accumulated = (
                    existing.degree_days_accumulated * existing.abundance + g.degree_days_accumulated * g.abundance
                ) / tot
            existing.abundance = tot
            existing.count = int(round(tot))
    return list(merged.values())


def perform_development(
    organism: Organism,
    params: IBMParameters,
    temp: float,
    dt: float = 1.0,
    day_of_year: Optional[float] = None,
    current_time: float = 0.0,
    food_avail: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Organism:
    """Advance organism bioenergetics, degree-days, aging, and ecdysis-linked development."""
    d_base = params.environment.temp_base_degreedays
    organism.degree_days_accumulated += max(0.0, temp - d_base) * dt
    organism.age_days += dt
    organism.shell_age_days += dt

    if organism.is_soft_shell and organism.shell_age_days > params.mortality.soft_shell_duration:
        organism.is_soft_shell = False
        if organism.state == "molting":
            organism.state = "foraging"

    if check_molt_conditions(organism, params, temp, day_of_year=day_of_year, dt=dt, rng=rng):
        organism = perform_molt(
            organism, params, temp, current_time=current_time, rng=rng
        )

    energy_delta = calculate_energy_change(organism, params, temp, food_avail=food_avail)
    organism.energy_reserves += energy_delta * dt

    if organism.energy_reserves < 0.3 * params.organism.initial_energy:
        organism.state = "resting"
    elif organism.state == "resting" and organism.energy_reserves > 0.5 * params.organism.initial_energy:
        organism.state = "foraging"

    return organism


def perform_reproduction(organism: Organism, params: IBMParameters, rng: Optional[random.Random] = None) -> float:
    """Calculate reproductive output (fecundity) for a mature female."""
    if not params.toggles.use_reproduction:
        return 0.0
    if organism.sex != Sex.FEMALE or not organism.is_mature:
        return 0.0
    return default_species_behavior().fecundity(organism, params, rng=rng)


def is_empirically_measured(organism: Organism) -> bool:
    """Check if organism belongs to empirically measured benthic fraction (stage >= 11)."""
    return organism.molt_stage >= 11 or organism.stage == LifeStage.ADULT


def empirical_measured_abundance(model) -> int:
    """Count organisms in the empirically measured benthic fraction."""
    return sum(1 for a in model.agents if is_empirically_measured(a.organism))


def benthic_carrying_capacity(
    params: IBMParameters,
    grid_size: Optional[Tuple[int, int]] = None,
    unit_meters: Optional[float] = None,
) -> float:
    """Derive benthic carrying capacity from density and physical domain area."""
    env = params.environment
    sim = params.simulation
    gs = grid_size if grid_size is not None else (sim.grid_nx, sim.grid_ny)
    u_m = float(unit_meters if unit_meters is not None else sim.unit_meters)
    area_km2 = (gs[0] * u_m) * (gs[1] * u_m) / 1e6
    if area_km2 >= 0.1 and env.benthic_carrying_capacity_density > 0.0:
        return env.benthic_carrying_capacity_density * area_km2
    return env.benthic_carrying_capacity


def benthic_density_feedback(density: float, params: IBMParameters) -> float:
    """Calculate density-dependent baseline mortality multiplier."""
    k = benthic_carrying_capacity(params)
    if k <= 0.0:
        return 1.0
    return 1.0 + (max(0.0, density) / k) ** params.mortality.density_feedback_gamma


def total_benthic_abundance(model) -> int:
    return len(model.agents)


def step_aggregate_groups(
    groups: List[AggregateGroup],
    params: IBMParameters,
    temp: float,
    dt: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Tuple[List[AggregateGroup], List[Tuple[int, float, Tuple[int, int]]]]:
    """Advance aggregate pelagic/juvenile cohorts; returns (surviving_groups, settled_recruits)."""
    r = rng if rng is not None else random
    m = params.mortality
    d_base = params.environment.temp_base_degreedays
    juv_max = params.species.juvenile_max_stage

    updated: List[AggregateGroup] = []
    settled: List[Tuple[int, float, Tuple[int, int]]] = []

    for g in groups:
        if g.abundance <= 0.0:
            continue
        g.age_days += dt
        g.degree_days_accumulated += max(0.0, temp - d_base) * dt

        # Stage-specific baseline + thermal larval mortality
        b_rates = m.aggregate_baseline_hazards
        h_base = b_rates[min(g.stage_idx, len(b_rates) - 1)]
        if g.stage_idx <= 3:
            t_pen = 0.02 * (1.0 - math.exp(-0.5 * ((temp - m.larval_temp_opt) / max(0.1, m.larval_temp_sigma)) ** 2))
            h_base += t_pen
            if g.age_days > 84.0:
                h_base += m.larval_overage_hazard
        surv_prob = math.exp(-max(0.0, h_base) * dt)
        g.abundance *= surv_prob
        if g.abundance < 0.5:
            continue

        # Poisson-CDF stage transition
        req_dd = required_degree_days(g.stage_idx, params)
        req_days = min_intermolt_days(g.stage_idx, params)
        p_dd = _poisson_cdf(int(math.floor(g.degree_days_accumulated)), max(1.0, req_dd))
        p_age = _poisson_cdf(int(math.floor(g.age_days)), max(1.0, req_days))
        trans_prob = 1.0 - math.exp(-m.aggregate_transition_hazard * min(p_dd, p_age) * dt)
        n_trans = int(round(g.abundance * trans_prob))

        if n_trans > 0:
            g.abundance = max(0.0, g.abundance - n_trans)
            next_s = g.stage_idx + 1
            if next_s >= juv_max:
                settled.append((n_trans, get_stage_size(next_s, params, Sex.MALE, mode="mid"), g.pos))
            else:
                stage_enum = LifeStage.ZOEA if next_s <= 2 else (LifeStage.MEGALOPA if next_s == 3 else LifeStage.JUVENILE)
                updated.append(
                    AggregateGroup(
                        stage=stage_enum,
                        count=n_trans,
                        mean_size=get_stage_size(next_s, params, Sex.MALE, mode="mid"),
                        age_days=0.0,
                        degree_days_accumulated=0.0,
                        stage_idx=next_s,
                        abundance=float(n_trans),
                        pos=g.pos,
                    )
                )
        if g.abundance >= 0.5:
            g.count = int(round(g.abundance))
            updated.append(g)

    return updated, settled


def calculate_habitat_suitability(temp: float, depth: float, params: Optional[IBMParameters] = None) -> float:
    """Compute Habitat Suitability Index (0-1) matching TethysBio::calculate_habitat_suitability."""
    # Thermal suitability: decreasing logistic around 5.0 C with cosine taper to 10.0 C
    if temp >= 10.0:
        s_temp = 0.0
    else:
        s_temp = 1.0 / (1.0 + math.exp(1.2 * (temp - 5.0)))
        if temp > 5.0:
            s_temp *= 0.5 * (1.0 + math.cos(math.pi * (temp - 5.0) / 5.0))

    # Depth suitability: Gaussian centered at 140 m with shallow (20 m) and deep (400 m) tapers
    if depth <= 20.0 or depth >= 400.0:
        s_depth = 0.0
    else:
        s_depth = math.exp(-0.5 * ((depth - 140.0) / 90.0) ** 2)
        if depth < 50.0:
            s_depth *= 0.5 * (1.0 - math.cos(math.pi * (depth - 20.0) / 30.0))
        elif depth > 300.0:
            s_depth *= 0.5 * (1.0 + math.cos(math.pi * (depth - 300.0) / 100.0))

    hsi = s_temp * s_depth
    if params is not None:
        hsi = params.environment.habitat_offset + params.environment.habitat_multiplier * hsi
    return max(0.0, min(1.0, hsi))
