"""Species behavior and allometric functions for AgentCrab.

Snow crab (Chionoecetes opilio) biological rules:
- Sexual dimorphism: males grow larger (max 160mm) than females (max 95mm)
- Maturity ogives differ by sex: male L50=91.8mm, female L50=51.6mm
- Fecundity: primiparous (first brood) vs multiparous (subsequent broods)
- Growth: sex-specific Hiatt logarithmic growth with maturity penalty
- Terminal morphometric maturity is irreversible
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from .types import IBMParameters, LifeStage, Organism, Sex


class SpeciesBehavior(Protocol):
    """Protocol for species-specific biological behavior."""

    def stage_size_lower_bound(self, stage: LifeStage, params: IBMParameters, sex: Sex = Sex.MALE) -> float: ...
    def growth_increment(self, organism: Organism, params: IBMParameters, temp: float) -> float: ...
    def maturity_probability(self, organism: Organism, params: IBMParameters, temp: float) -> float: ...
    def fecundity(self, organism: Organism, params: IBMParameters) -> float: ...
    def max_size(self, sex: Sex, params: IBMParameters) -> float: ...


@dataclass
class SpeciesBehaviorImpl:
    """Default implementation of SpeciesBehavior for snow crab."""

    def max_size(self, sex: Sex, params: IBMParameters) -> float:
        sp = params.species
        if sex == Sex.FEMALE:
            return sp.max_size_female
        return sp.max_size_male

    def stage_size_lower_bound(self, stage: LifeStage, params: IBMParameters, sex: Sex = Sex.MALE) -> float:
        sp = params.species
        max_sz = sp.max_size_female if sex == Sex.FEMALE else sp.max_size_male
        ln_s0 = math.log(sp.stage_lower_bound_s0)
        ln_linf = math.log(max_sz)
        kappa = sp.stage_lower_bound_kappa
        iota0 = sp.stage_lower_bound_iota0
        iota = 0.0
        return math.exp(ln_linf + (ln_s0 - ln_linf) * math.exp(-kappa * (iota - iota0)))

    def growth_increment(self, organism: Organism, params: IBMParameters, temp: float) -> float:
        sp = params.species
        if organism.sex == Sex.FEMALE:
            beta0 = sp.growth_beta0_female
            beta1 = sp.growth_beta1_female
            delta_mat = sp.growth_maturity_reduction_female
            max_size = sp.max_size_female
        else:
            beta0 = sp.growth_beta0_male
            beta1 = sp.growth_beta1_male
            delta_mat = sp.growth_maturity_reduction_male
            max_size = sp.max_size_male

        gamma_t = sp.growth_temp_coeff
        t_ref = sp.growth_ref_temp
        is_mat = 1.0 if organism.is_mature else 0.0
        cw = max(1.0, organism.carapace_width)

        ln_cw = math.log(cw)
        ln_next = beta0 + beta1 * ln_cw + delta_mat * is_mat + gamma_t * (temp - t_ref)
        next_cw = math.exp(ln_next)

        return min(next_cw, max_size) - cw

    def maturity_probability(self, organism: Organism, params: IBMParameters, temp: float) -> float:
        sp = params.species
        if organism.sex == Sex.FEMALE:
            l50 = sp.cw50_maturity_female
            slope = sp.maturity_slope_female
        else:
            l50 = sp.cw50_maturity
            slope = sp.maturity_slope_male

        size_gate = sp.maturity_size_gate
        if organism.carapace_width < size_gate:
            return 0.0

        alpha = -slope * math.log(l50)
        gamma_t = sp.growth_temp_coeff
        t_ref = sp.growth_ref_temp
        logit_val = alpha + slope * math.log(max(1.0, organism.carapace_width)) + gamma_t * (temp - t_ref)
        return 1.0 / (1.0 + math.exp(-logit_val))

    def fecundity(self, organism: Organism, params: IBMParameters) -> float:
        sp = params.species
        if not organism.is_adult_female() or not organism.is_mature:
            return 0.0

        cw = organism.carapace_width
        shell_age_years = organism.shell_age_days / 365.0

        # Primiparous vs multiparous based on shell age
        if shell_age_years <= sp.primiparous_shell_age_max:
            intercept = sp.fecundity_primiparous_intercept
            slope = sp.fecundity_primiparous_slope
        else:
            intercept = sp.fecundity_multiparous_intercept
            slope = sp.fecundity_multiparous_slope

        log_eggs = intercept + slope * math.log(max(1.0, cw))
        eggs = math.exp(log_eggs)

        # Senescence discount for older females
        if shell_age_years > sp.fecundity_senescence_threshold:
            discount = max(sp.fecundity_senescence_floor,
                           1.0 - sp.fecundity_senescence_rate * (shell_age_years - sp.fecundity_senescence_threshold))
            eggs *= discount

        return max(1000.0, eggs)


def default_species_behavior() -> SpeciesBehaviorImpl:
    return SpeciesBehaviorImpl()


# ── Allometric helpers ────────────────────────────────────────────────


def max_size(sex: Sex, params: IBMParameters) -> float:
    sp = params.species
    if sex == Sex.FEMALE:
        return sp.max_size_female
    return sp.max_size_male


def calculate_mass(cw: float, sex: Sex, is_mature: bool, params: IBMParameters) -> float:
    """Compute mass (g) from carapace width using sex-specific allometry."""
    sp = params.species
    if sex == Sex.FEMALE:
        alpha = sp.mass_alpha_female_mature if is_mature else sp.mass_alpha_female_immature
        beta = sp.mass_beta_female_mature if is_mature else sp.mass_beta_female_immature
    else:
        alpha = sp.mass_alpha_male_mature if is_mature else sp.mass_alpha_male_immature
        beta = sp.mass_beta_male_mature if is_mature else sp.mass_beta_male_immature
    return alpha * (cw ** beta)


def llbm(cw: float, params: IBMParameters) -> float:
    """Lower size bound for male organisms."""
    sp = params.species
    return sp.asymptotic_size * (1.0 - math.exp(-sp.stage_lower_bound_kappa * cw))


def llbf(cw: float, params: IBMParameters) -> float:
    """Lower size bound for female organisms (smaller than males)."""
    sp = params.species
    return sp.max_size_female * (1.0 - math.exp(-sp.stage_lower_bound_kappa * cw))


def get_stage_size(stage: LifeStage, params: IBMParameters, sex: Sex = Sex.MALE) -> float:
    """Typical size for a given life stage, sex-aware."""
    sp = params.species
    if stage == LifeStage.ZOEA:
        return 2.0
    elif stage == LifeStage.MEGALOPA:
        return 5.0
    elif stage == LifeStage.JUVENILE:
        return 20.0
    elif stage == LifeStage.ADULT:
        if sex == Sex.FEMALE:
            return sp.cw50_maturity_female
        return sp.cw50_maturity
    return 1.0


def maturity_probability(organism, params: IBMParameters, temp: float) -> float:
    return default_species_behavior().maturity_probability(organism, params, temp)


def predict_postgrowth_size(cw: float, params: IBMParameters, temp: float, sex: Sex = Sex.MALE) -> float:
    """Predict carapace width after a molt, sex-aware."""
    sp = params.species
    if sex == Sex.FEMALE:
        beta0 = sp.growth_beta0_female
        beta1 = sp.growth_beta1_female
        max_size = sp.max_size_female
    else:
        beta0 = sp.growth_beta0_male
        beta1 = sp.growth_beta1_male
        max_size = sp.max_size_male

    ln_next = beta0 + beta1 * math.log(max(1.0, cw))
    next_cw = math.exp(ln_next)
    return min(next_cw, max_size)


def build_stages(params: IBMParameters) -> list[LifeStage]:
    """Build list of life stages for the species."""
    return [LifeStage.ZOEA, LifeStage.MEGALOPA, LifeStage.JUVENILE, LifeStage.ADULT]
