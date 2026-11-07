"""Organism dynamics and bioenergetics for AgentCrab."""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

from .types import (
    AggregateGroup,
    IBMParameters,
    LifeStage,
    Organism,
    ProcessToggles,
    Sex,
)
from .species import (
    default_species_behavior,
    get_stage_size,
    predict_postgrowth_size,
    maturity_probability,
)


def calculate_energy_change(organism: Organism, params: IBMParameters, temp: float) -> float:
    """Calculate daily energy change for an organism."""
    if not params.toggles.bioenergetics:
        return 0.0

    cw = organism.carapace_width
    mass = organism.mass

    # Temperature-dependent metabolic rate (Q10)
    q10 = params.organism.metabolic_q10
    t_ref = params.species.growth_ref_temp
    metabolic_rate = mass * params.organism.metabolic_rate_base * (q10 ** ((temp - t_ref) / 10.0))

    # Assimilation (simplified: proportional to mass)
    assimilation = mass * params.organism.intake_rate_base

    return assimilation - metabolic_rate


def check_molt_conditions(organism: Organism, params: IBMParameters, temp: float) -> bool:
    """Check if organism is ready to molt."""
    if not params.toggles.molting:
        return False
    if organism.degree_days_accumulated < params.phenology.required_degree_days:
        return False
    if organism.shell_age_days < params.phenology.min_intermolt_days:
        return False
    return random.random() < params.phenology.molting_probability_base


def perform_molt(organism: Organism, params: IBMParameters, temp: float) -> Organism:
    """Perform molting: update size, reset shell age."""
    from .species import calculate_mass, predict_postgrowth_size

    new_cw = predict_postgrowth_size(organism.carapace_width, params, temp, organism.sex)
    organism.carapace_width = new_cw
    organism.shell_age_days = 0.0
    organism.degree_days_accumulated = 0.0
    organism.is_soft_shell = True
    organism.state = "molting"
    organism.mass = calculate_mass(new_cw, organism.sex, organism.is_mature, params)
    return organism


def molt_probability(organism: Organism, params: IBMParameters) -> float:
    """Calculate probability of molting."""
    if not params.toggles.molting:
        return 0.0
    if organism.degree_days_accumulated < params.phenology.required_degree_days:
        return 0.0
    if organism.shell_age_days < params.phenology.min_intermolt_days:
        return 0.0
    return params.phenology.molting_probability_base


def calculate_mortality_probability(organism: Organism, params: IBMParameters,
                                     temp: float, density: float) -> float:
    """Calculate daily mortality probability using all hazard components."""
    m = params.mortality
    cw = organism.carapace_width
    mass = organism.mass

    # Base mortality (stage-specific)
    baseline_rates = m.baseline_rate_by_stage
    # Map LifeStage enum to baseline_rate_by_stage index
    stage_to_idx = {
        LifeStage.EGG: 0,
        LifeStage.ZOEA: 1,
        LifeStage.MEGALOPA: 2,
        LifeStage.JUVENILE: 3,
        LifeStage.ADULT: 4,
    }
    stage_idx = stage_to_idx.get(organism.stage, 0)
    if stage_idx < len(baseline_rates):
        m_base = baseline_rates[stage_idx] / 365.0
    else:
        m_base = baseline_rates[-1] / 365.0

    # Predation (size-dependent)
    m_pred = 0.0
    if cw > 0:
        m_pred = m.predation_coeff / max(1.0, cw - m.predation_size_reference + 1.0)
        m_pred = max(0.0, m_pred)

    # Soft shell vulnerability
    m_soft = 0.0
    if organism.is_soft_shell:
        m_soft = (m.soft_shell_mortality / 365.0)
        if organism.shell_age_days > m.soft_shell_duration:
            m_soft = 0.0
        else:
            m_soft *= math.exp(-organism.shell_age_days / m.soft_shell_decay_days)

    # Starvation (energy-dependent)
    m_starve = 0.0
    if organism.energy_reserves < m.starvation_critical_fraction:
        m_starve = m.starvation_threshold * (1.0 - organism.energy_reserves / m.starvation_critical_fraction)

    # Thermal stress
    m_thermal = 0.0
    if params.toggles.thermal_stress:
        t_stress = m.temp_thermal_stress
        if temp > t_stress:
            excess = temp - t_stress
            m_thermal = m.thermal_stress_coeff * (excess ** m.thermal_exponent) / 10.0

    # Senescence (age-dependent, unbounded, capped)
    m_senescence = 0.0
    if params.toggles.adult_senescence and organism.is_adult():
        shell_age_years = organism.shell_age_days / 365.0
        if shell_age_years > m.senescence_age_threshold:
            m_senescence = m.rate_senescence * math.exp(m.senescence_exp_rate * (shell_age_years - m.senescence_age_threshold))
            m_senescence = min(m_senescence, m.hazard_cap)

    # Cannibalism (density-dependent)
    m_cannibal = 0.0
    if params.toggles.cannibalism:
        if organism.carapace_width >= m.cannibalism_size_threshold:
            density_ratio = density / m.cannibalism_density_reference
            m_cannibal = m.cannibalism_coeff * min(density_ratio, m.cannibalism_density_cap)

    # Total mortality (capped at 1.0)
    total = m_base + m_pred + m_soft + m_starve + m_thermal + m_senescence + m_cannibal

    # Density feedback
    k = params.environment.benthic_carrying_capacity
    if k > 0:
        total *= benthic_density_feedback(density, params)

    return min(1.0, total)


def mortality_components(organism: Organism, params: IBMParameters,
                          temp: float, density: float) -> dict:
    """Return individual mortality components for diagnostics."""
    m = params.mortality
    cw = organism.carapace_width
    return {
        "base": m.base_mortality,
        "predation": m.predation_coeff / max(1.0, cw),
        "soft_shell": m.soft_shell_mortality if organism.is_soft_shell else 0.0,
        "starvation": m.starvation_threshold * (1.0 - organism.energy_reserves) if organism.energy_reserves < 0.5 else 0.0,
        "thermal": m.thermal_stress_coeff * abs(temp - params.species.growth_ref_temp) / 10.0 if params.toggles.thermal_stress else 0.0,
        "senescence": 0.0,
        "cannibalism": m.cannibalism_coeff * density if params.toggles.cannibalism else 0.0,
    }


def shell_age_years(organism: Organism) -> float:
    return organism.shell_age_days / 365.0


def shell_condition(organism: Organism) -> str:
    if organism.is_soft_shell:
        return "soft"
    if organism.shell_age_days < 30:
        return "new"
    return "hard"


def min_intermolt_days(organism: Organism, params: IBMParameters) -> int:
    return params.phenology.min_intermolt_days


def required_degree_days(organism: Organism, params: IBMParameters) -> float:
    return params.phenology.required_degree_days


def perform_development(organism: Organism, params: IBMParameters, temp: float) -> Organism:
    """Advance organism development (molting, maturity)."""
    from .species import maturity_probability, predict_postgrowth_size

    # Accumulate degree days
    organism.degree_days_accumulated += max(0.0, temp)

    # Check molt
    if check_molt_conditions(organism, params, temp):
        organism = perform_molt(organism, params, temp)

    # Check maturity
    if not organism.is_mature and organism.is_adult():
        prob = maturity_probability(organism, params, temp)
        if random.random() < prob:
            organism.is_mature = True

    # Age
    organism.age_days += 1.0
    organism.shell_age_days += 1.0

    # State transitions
    if organism.state == "molting" and not organism.is_soft_shell:
        # Just completed a molt
        organism.state = "foraging"
    elif organism.state == "foraging" and organism.energy_reserves < 0.3:
        organism.state = "resting"
    elif organism.state == "resting" and organism.energy_reserves > 0.5:
        organism.state = "foraging"
    elif organism.state == "foraging" and organism.energy_reserves >= 0.3:
        organism.state = "foraging"
    elif organism.state == "resting" and organism.energy_reserves <= 0.5:
        organism.state = "resting"

    # Energy update
    energy_change = calculate_energy_change(organism, params, temp)
    organism.energy_reserves = max(0.0, min(1.0, organism.energy_reserves + energy_change))

    return organism


def perform_reproduction(organism: Organism, params: IBMParameters) -> float:
    """Calculate reproductive output (fecundity) for an organism."""
    if not params.toggles.reproduction:
        return 0.0
    if not organism.is_adult_female() or not organism.is_mature:
        return 0.0

    from .species import default_species_behavior
    return default_species_behavior().fecundity(organism, params)


def benthic_density_feedback(density: float, params: IBMParameters) -> float:
    """Calculate density-dependent feedback factor."""
    k = params.environment.benthic_carrying_capacity
    if k <= 0:
        return 1.0
    gamma = params.mortality.density_feedback_gamma
    return 1.0 + (density / k) ** gamma


def total_benthic_abundance(model) -> int:
    """Count total benthic organisms in model."""
    return len(model.agents)


def step_aggregate_groups(groups: list, params: IBMParameters,
                          temp: float, dt: float = 1.0) -> list:
    """Advance aggregate cohort groups by one time step."""
    new_groups = []
    for g in groups:
        g.age_days += dt
        g.degree_days_accumulated += max(0.0, temp) * dt
        if g.degree_days_accumulated >= params.phenology.required_degree_days:
            g.degree_days_accumulated = 0.0
        new_groups.append(g)
    return new_groups


def calculate_habitat_suitability(temp: float, depth: float, params: IBMParameters) -> float:
    """Calculate habitat suitability index (0-1)."""
    t_opt = 2.0
    t_tol = 5.0
    suitability = math.exp(-((temp - t_opt) / t_tol) ** 2)
    if depth < 50:
        suitability *= 0.5
    elif depth > 500:
        suitability *= 0.3
    return max(0.0, min(1.0, suitability))
