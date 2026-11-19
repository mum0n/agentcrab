"""Core types for AgentCrab individual-based simulation model."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


class Sex(enum.Enum):
    MALE = "male"
    FEMALE = "female"
    UNDIFFERENTIATED = "undifferentiated"


class LifeStage(enum.Enum):
    EGG = "egg"
    ZOEA = "zoea"
    MEGALOPA = "megalopa"
    JUVENILE = "juvenile"
    ADULT = "adult"


# ── Parameter structures ──────────────────────────────────────────────


@dataclass
class FisheryParams:
    fishing_probability_daily: float = 0.0
    legal_size_male: float = 95.0
    fishing_season_start: float = 90.0
    fishing_season_end: float = 150.0
    no_take_proportion: float = 0.0

    @property
    def legal_size_mm(self) -> float:
        return self.legal_size_male

    @legal_size_mm.setter
    def legal_size_mm(self, val: float) -> None:
        self.legal_size_male = float(val)


@dataclass
class EnvironmentParams:
    temp_offset: float = 0.0
    temp_base: float = 0.0
    temp_multiplier: float = 1.0
    temp_base_degreedays: float = -1.5
    benthic_carrying_capacity: float = 6000.0
    benthic_carrying_capacity_density: float = 6000.0
    benthic_carrying_capacity_exponent: float = 2.0
    food_availability: float = 1.0
    half_saturation_food: float = 2.0
    assimilation_efficiency: float = 0.75
    habitat_quality_mean: float = 0.5
    habitat_quality_sd: float = 0.1
    habitat_offset: float = 0.0
    habitat_multiplier: float = 1.0
    habitat_temperature_optimum: float = 2.5
    habitat_temperature_sigma: float = 1.5
    habitat_food_weight: float = 0.5
    movement_softmax_temperature: float = 0.5
    food_max: float = 10.0
    food_replenishment_rate: float = 0.1
    grid_size: Tuple[int, int] = (100, 100)
    use_geodata: bool = False
    domain_lon: list = field(default_factory=lambda: [-65.0, -63.0])
    domain_lat: list = field(default_factory=lambda: [43.0, 45.0])
    time_range: list = field(default_factory=lambda: ["1993-01-01", "1993-12-31"])
    pelagic_depth: float = 15.0
    temperature_source: str = "woa23"
    temperature_month: int = 0
    bathymetry_source: str = "nwa_bathymetry"
    geodata_input_dir: str = "inputs"
    geodata_store_dir: str = "data/environmental"
    settlement_dispersal_radius: int = 2
    settlement_size_log_sd: float = 0.04


@dataclass
class SpeciesParams:
    name: str = "snowcrab"
    max_size_male: float = 160.0
    max_size_female: float = 95.0
    cw50_maturity: float = 91.8
    cw50_maturity_female: float = 51.6
    maturity_slope_male: float = 6.9957
    maturity_slope_female: float = 12.9812
    maturity_size_gate: float = 35.0
    maturity_l50_male_sd: float = 1.2
    maturity_l50_female_sd: float = 0.8
    maturity_slope_male_sd: float = 0.25
    maturity_slope_female_sd: float = 0.4
    maturity_l50_temp_slope: float = -2.0
    maturity_temp_reference: float = 4.0
    growth_beta0_male: float = 0.30153
    growth_beta0_female: float = 0.1931
    growth_maturity_reduction_male: float = -0.13564
    growth_maturity_reduction_female: float = -0.131
    growth_temp_coeff: float = -0.004
    growth_ref_temp: float = 4.0
    growth_noise_sd: float = 0.04
    asymptotic_size: float = 160.0
    stage_lower_bound_kappa: float = 0.235833
    stage_lower_bound_s0: float = 1.102422
    stage_lower_bound_iota0: float = 4.0
    fecundity_parameterization: str = "moncton"
    fecundity_primiparous_intercept: float = -0.7229
    fecundity_primiparous_slope: float = 2.94
    fecundity_primiparous_err: float = 0.09
    fecundity_multiparous_intercept: float = 0.0158
    fecundity_multiparous_slope: float = 2.5811
    fecundity_multiparous_err: float = 0.145
    fecundity_intercept_sd: float = 0.05
    fecundity_senescence_threshold: float = 3.0
    fecundity_senescence_floor: float = 0.6
    fecundity_senescence_rate: float = 0.08
    primiparous_shell_age_max: float = 2.0
    fecundity_parity_size_split: float = 65.0
    fecundity_stmarie_exponent: float = 2.725
    fecundity_stmarie_coefficient: float = -0.7311
    # Growth (TethysBio Hiatt + individual quantile parameterization)
    growth_int_female: float = 0.1931
    growth_beta1_female: float = 0.0545
    growth_slp_female: float = 1.0372
    growth_beta2_female: float = -0.017376
    growth_beta3_female: float = 0.005068
    growth_int_male: float = 0.30153
    growth_beta1_male: float = 0.02198
    growth_slp_male: float = 1.00027
    growth_beta2_male: float = -0.00643655
    growth_beta3_male: float = 0.0017492
    growth_maturation_penalty_male_mean: float = -0.13564
    growth_maturation_penalty_male_sd: float = 0.00922
    growth_maturation_penalty_female_mean: float = -0.131
    growth_maturation_penalty_female_sd: float = 0.0252
    growth_condition_coefficient: float = 0.05
    growth_temperature_coefficient: float = -0.004
    growth_temperature_reference: float = 4.0
    growth_residual_sd: float = 0.04
    growth_min_expansion_immature: float = 0.12
    growth_min_expansion_mature: float = 0.06
    # Stage size lower-bound saturation curves
    size_lb_immature_sizes: list = field(default_factory=lambda: [0.95, 1.45, 2.5])
    size_lb_immature_breakpoints: list = field(default_factory=lambda: [1.5, 2.5])
    size_lb_male_intercept: float = 1.102422
    size_lb_male_decay: float = 0.235833
    size_lb_male_anchor_stage: float = 4.0
    size_lb_female_intercept: float = 1.930054
    size_lb_female_decay: float = 0.308476
    size_lb_female_anchor_stage: float = 5.0
    benthic_anchor_stage: float = 5.0
    # Stage size ranges
    max_stage_male: int = 19
    max_stage_female: int = 16
    juvenile_max_stage: int = 12
    adult_size_threshold: float = 68.0
    # Wet-mass allometry: mass = alpha * size^beta (g)
    mass_alpha_male_immature: float = 0.000183
    mass_beta_male_immature: float = 3.16
    mass_alpha_male_mature: float = 0.0002665
    mass_beta_male_mature: float = 3.098
    mass_alpha_female_immature: float = 0.000344
    mass_beta_female_immature: float = 2.98
    mass_alpha_female_mature: float = 0.00045
    mass_beta_female_mature: float = 2.92
    mass_sigma: float = 0.08
    # Selectivity
    selectivity_s50: float = 95.0
    selectivity_s50_sigma: float = 5.0
    # Stage labels
    stage_labels: list = field(default_factory=lambda: [
        "Egg", "z1", "z2", "m", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "11", "12", "13", "14", "15", "16"
    ])
    # Recruitment
    recruitment_stage: int = 13
    # Stage size ranges (lower/upper/mean per stage)
    stage_size_range_stage1: list = field(default_factory=lambda: [7.0, 8.0])
    stage_size_range_lower_stage1: list = field(default_factory=lambda: [7.0, 14.5])
    stage_size_range_upper_stage1: list = field(default_factory=lambda: [14.5, 22.0])
    stage_size_range_mean_stage1: list = field(default_factory=lambda: [10.7, 18.0])
    stage_size_range_stage_male: list = field(default_factory=lambda: [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    stage_size_range_lower_male: list = field(default_factory=lambda: [22.0, 31.0, 42.5, 58.0, 78.0, 95.0, 115.0])
    stage_size_range_upper_male: list = field(default_factory=lambda: [31.0, 42.5, 58.0, 78.0, 95.0, 115.0, 160.0])
    stage_size_range_mean_male: list = field(default_factory=lambda: [26.5, 36.5, 50.0, 68.0, 86.5, 105.0, 130.0])
    stage_size_range_stage_female: list = field(default_factory=lambda: [9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    stage_size_range_lower_female: list = field(default_factory=lambda: [22.0, 30.5, 41.5, 54.5, 68.0, 82.0])
    stage_size_range_upper_female: list = field(default_factory=lambda: [30.5, 41.5, 54.5, 68.0, 82.0, 110.0])
    stage_size_range_mean_female: list = field(default_factory=lambda: [26.0, 35.5, 47.5, 61.0, 75.0, 90.0])
    leslie_max_male_stage: int = 15
    leslie_max_female_stage: int = 14
    leslie_legal_size_male: float = 95.0
    leslie_first_legal_stage_male: int = 14
    leslie_first_repro_stage_female: int = 12
    # Intermolt schedules
    intermolt_days_schedule: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 182.5, 182.5, 182.5, 182.5, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0, 365.0])
    degree_days_schedule: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 912.5, 912.5, 912.5, 912.5, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0, 1825.0])
    intermolt_days_larval: list = field(default_factory=lambda: [0.0, 36.0, 36.0, 42.0])
    degree_days_larval: list = field(default_factory=lambda: [0.0, 270.0, 270.0, 270.0])


@dataclass
class ProcessToggles:
    use_energetics: bool = True
    use_thermal_stress: bool = True
    use_cannibalism: bool = False
    use_senescence: bool = True
    use_spermatheca: bool = True
    use_movement: bool = True
    use_directed_movement: bool = True
    use_degree_days: bool = True
    use_molting: bool = True
    use_reproduction: bool = True

    @property
    def bioenergetics(self) -> bool:
        return self.use_energetics

    @bioenergetics.setter
    def bioenergetics(self, v: bool) -> None:
        self.use_energetics = bool(v)

    @property
    def thermal_stress(self) -> bool:
        return self.use_thermal_stress

    @thermal_stress.setter
    def thermal_stress(self, v: bool) -> None:
        self.use_thermal_stress = bool(v)

    @property
    def cannibalism(self) -> bool:
        return self.use_cannibalism

    @cannibalism.setter
    def cannibalism(self, v: bool) -> None:
        self.use_cannibalism = bool(v)

    @property
    def adult_senescence(self) -> bool:
        return self.use_senescence

    @adult_senescence.setter
    def adult_senescence(self, v: bool) -> None:
        self.use_senescence = bool(v)

    @property
    def spermathecal_storage(self) -> bool:
        return self.use_spermatheca

    @spermathecal_storage.setter
    def spermathecal_storage(self, v: bool) -> None:
        self.use_spermatheca = bool(v)

    @property
    def directed_habitat_taxis(self) -> bool:
        return self.use_directed_movement

    @directed_habitat_taxis.setter
    def directed_habitat_taxis(self, v: bool) -> None:
        self.use_directed_movement = bool(v)

    @property
    def degree_day_ecdysial_gating(self) -> bool:
        return self.use_degree_days

    @degree_day_ecdysial_gating.setter
    def degree_day_ecdysial_gating(self, v: bool) -> None:
        self.use_degree_days = bool(v)

    @property
    def molting(self) -> bool:
        return self.use_molting

    @molting.setter
    def molting(self, v: bool) -> None:
        self.use_molting = bool(v)

    @property
    def reproduction(self) -> bool:
        return self.use_reproduction

    @reproduction.setter
    def reproduction(self, v: bool) -> None:
        self.use_reproduction = bool(v)


@dataclass
class PhenologyParams:
    min_intermolt_days: int = 30
    required_degree_days: float = 500.0
    molting_probability_base: float = 0.1
    molt_season_start: float = 15.0
    molt_season_end: float = 140.0
    mating_season_start: float = 30.0
    mating_season_end: float = 120.0
    mating_search_radius: int = 5
    mating_energy_cost: float = 8.0
    mating_size_weight_exponent: float = 2.0
    sperm_stores_legal: int = 3
    sperm_stores_sublegal: int = 2
    minimum_clutch_size: int = 1000
    brood_duration_base: float = 365.0
    brood_temperature_reference: float = 5.0
    brood_temperature_exponent: float = 0.3
    brood_temperature_floor: float = 1.0
    brood_duration_sd: float = 5.0
    molt_season_min_stage: int = 13
    molt_thermal_onset: float = 10.0
    molt_thermal_bandwidth: float = 3.0
    shell_age_stage_offset: float = 180.0
    shell_age_first_stage: int = 6
    shell_condition_days: list = field(default_factory=lambda: [30.0, 365.0, 730.0, 1277.5])
    egg_mass_g: float = 7.0e-5
    egg_size_mm: float = 0.6
    energy_deficit_removal_probability: float = 0.75


@dataclass
class MortalityParams:
    base_mortality: float = 0.001
    predation_coeff: float = 0.05
    soft_shell_mortality: float = 0.12
    starvation_threshold: float = 0.05
    thermal_stress_coeff: float = 0.0
    senescence_age: float = 3.0
    cannibalism_coeff: float = 0.0
    density_feedback_gamma: float = 2.0
    # Stage-specific baseline rates
    baseline_rate_by_stage: list = field(default_factory=lambda: [0.35, 0.35, 0.35, 0.35, 0.15, 0.15, 0.15, 0.15, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.06])
    rate_predation: float = 0.05
    rate_soft_shell: float = 0.12
    rate_starvation: float = 0.2
    rate_thermal: float = 0.05
    rate_senescence: float = 0.05
    rate_cannibalism: float = 0.05
    baseline_trait_sigma: float = 0.25
    rate_handling: float = 0.05
    handling_mortality_days: float = 365.0
    predation_size_reference: float = 5.0
    predation_size_decay: float = 0.03
    soft_shell_duration: float = 30.0
    soft_shell_decay_days: float = 15.0
    starvation_critical_fraction: float = 0.05
    temp_thermal_stress: float = 8.0
    thermal_exponent: float = 1.5
    senescence_age_threshold: float = 3.0
    senescence_exp_rate: float = 0.8
    cannibalism_size_threshold: float = 35.0
    cannibalism_density_reference: float = 10.0
    cannibalism_density_cap: float = 2.0
    aggregate_transition_hazard: float = 0.35
    aggregate_baseline_hazards: list = field(default_factory=lambda: [0.002, 0.020, 0.020, 0.015, 0.0015])
    larval_overage_hazard: float = 0.05
    larval_temp_opt: float = 11.0
    larval_temp_sigma: float = 3.5
    hazard_cap: float = 1.0
    # Energy deficit
    energy_deficit_removal_probability: float = 0.75


@dataclass
class SimulationParams:
    dt: float = 1.0
    seed: int = 1234
    days: int = 730
    days_per_year: float = 365.0
    temp_seasonal_phase_day: float = 80.0
    grid_nx: int = 1000
    grid_ny: int = 1000
    unit_meters: float = 1.0
    initial_aggregate_cohort: list = field(default_factory=lambda: [0.0, 2000.0] + [0.0] * 10)
    initial_female_cohort: list = field(default_factory=lambda: [0.0] * 20)
    initial_male_cohort: list = field(default_factory=lambda: [0.0] * 20)
    initial_cohort_days_per_stage: float = 90.0
    initial_age_days_per_stage: float = 180.0
    initial_age_first_stage: float = 3.0
    initial_age_jitter_days: float = 90.0
    initial_degree_days_min: float = 50.0
    initial_degree_days_max: float = 300.0
    n_agents: int = 2000


@dataclass
class LeslieParams:
    survival_stage1: float = 0.74
    gamma_stage1: float = 0.78
    survival_stage2: float = 0.76
    gamma_stage2: float = 0.75
    male_split_fraction: float = 0.5
    survival_male_legal: float = 0.78
    survival_male_sublegal: float = 0.8
    survival_female: float = 0.8
    gamma_male_base: float = 0.75
    gamma_male_decay: float = 0.1
    gamma_female_base: float = 0.7
    gamma_female_decay: float = 0.12
    p_term_male_base: float = 0.1
    p_term_male_exponent: float = 1.3
    p_term_female_base: float = 0.15
    p_term_female_exponent: float = 1.4
    fecundity_base: float = 0.85
    fecundity_size_exponent: float = 2.5
    fecundity_size_reference: float = 65.0
    trajectory_error_weight: float = 0.5
    first_sex_specific_stage: int = 9


@dataclass
class CriteriaParams:
    min_population: int = 1
    max_population_multiple: float = 3.0
    max_size_drift: float = 0.20
    max_maturity_rmse: float = 0.35
    max_growth_rmse: float = 15.0
    window_steps: int = 10
    maturity_ogive_min_count: int = 10
    extinction_penalty: float = 150.0
    explosion_penalty: float = 150.0
    size_drift_weight: float = 25.0
    maturity_rmse_weight: float = 15.0
    growth_rmse_weight: float = 1.0
    ogive_bin_start: float = 20.0
    ogive_bin_step: float = 10.0
    ogive_bin_stop: float = 140.0


@dataclass
class OutputParams:
    save: bool = True
    save_path: str = "simulation_results"
    snapshot_day: float = 274.0
    frequency_days: float = 365.0
    plot: bool = False
    output_dir: str = "outputs"
    animate: bool = False
    animation_file: str = "organism_animation.mp4"
    anim_frames: int = 120
    anim_dt: int = 2
    anim_fps: int = 15
    anim_field: str = "temperature"


@dataclass
class OrganismParams:
    metabolic_rate_base: float = 0.01
    intake_rate_base: float = 0.05
    metabolic_mass_exponent: float = 0.75
    metabolic_temperature_coefficient: float = 0.08
    metabolic_optimal_temperature: float = 3.0
    metabolic_optimal_sigma: float = 2.5
    metabolic_q10: float = 2.0
    specific_dynamic_action: float = 0.15
    energy_molt_slope: float = 0.03
    energy_molt_floor: float = 2.0
    molt_energy_retention: float = 0.1
    molt_energy_base_fraction: float = 0.85
    molt_hazard_coefficient: float = 0.15
    molt_progress_saturation: float = 3.0
    initial_energy: float = 10.0
    stage_size_sample_a: float = 2.5
    stage_size_sample_b: float = 2.5
    shell_age_stage_offset: float = 180.0
    shell_age_first_stage: int = 6
    egg_mass_g: float = 7.0e-5
    egg_size_mm: float = 0.6
    trait_quantile_min: float = 0.01
    trait_quantile_max: float = 0.99
    max_stage_male: int = 19
    max_stage_female: int = 16
    juvenile_max_stage: int = 12
    adult_size_threshold: float = 68.0
    stage_labels: list = field(default_factory=lambda: [
        "Egg", "z1", "z2", "m", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "11", "12", "13", "14", "15", "16"
    ])
    shell_condition_days: list = field(default_factory=lambda: [30.0, 365.0, 730.0, 1277.5])
    energy_deficit_removal_probability: float = 0.75


@dataclass
class IBMParameters:
    fishery: FisheryParams = field(default_factory=FisheryParams)
    environment: EnvironmentParams = field(default_factory=EnvironmentParams)
    species: SpeciesParams = field(default_factory=SpeciesParams)
    toggles: ProcessToggles = field(default_factory=ProcessToggles)
    phenology: PhenologyParams = field(default_factory=PhenologyParams)
    mortality: MortalityParams = field(default_factory=MortalityParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    leslie: LeslieParams = field(default_factory=LeslieParams)
    criteria: CriteriaParams = field(default_factory=CriteriaParams)
    output: OutputParams = field(default_factory=OutputParams)
    organism: OrganismParams = field(default_factory=OrganismParams)


# ── Organism agent ────────────────────────────────────────────────────


@dataclass
class Organism:
    unique_id: int
    sex: Sex
    stage: LifeStage
    carapace_width: float
    age_days: float = 0.0
    energy_reserves: float = 10.0
    molt_stage: int = 12
    degree_days_accumulated: float = 0.0
    is_mature: bool = False
    is_ovigerous: bool = False
    shell_age_days: float = 0.0
    is_soft_shell: bool = False
    spermatheca_stored: float = 0.0
    sperm_clutches: int = 0
    egg_count: float = 0.0
    hatch_day: float = -1.0
    is_mating_ready: bool = False
    last_growth_event_time: float = 0.0
    trait_growth_quantile: float = 0.5
    trait_maturity_quantile: float = 0.5
    trait_fecundity_quantile: float = 0.5
    trait_mortality_quantile: float = 0.5
    position: Tuple[int, int] = (0, 0)
    state: str = "foraging"
    mass: float = 0.0

    def __post_init__(self):
        """Compute mass from carapace width using sex-specific allometry when not explicitly set."""
        if self.molt_stage == 0:
            stage_map = {
                LifeStage.EGG: 0,
                LifeStage.ZOEA: 1,
                LifeStage.MEGALOPA: 3,
                LifeStage.JUVENILE: 8,
                LifeStage.ADULT: 13,
            }
            self.molt_stage = stage_map.get(self.stage, 12)
        if self.mass == 0.0:
            from .species import calculate_mass

            default_params = IBMParameters()
            self.mass = calculate_mass(
                self.carapace_width, self.sex, self.is_mature, default_params,
                include_eggs=self.is_ovigerous, egg_count=self.egg_count,
            )

    def is_adult(self) -> bool:
        return self.stage == LifeStage.ADULT

    def is_adult_male(self) -> bool:
        return self.sex == Sex.MALE and (self.stage == LifeStage.ADULT or self.molt_stage >= 12)

    def is_adult_female(self) -> bool:
        return self.sex == Sex.FEMALE and (self.stage == LifeStage.ADULT or self.molt_stage >= 12)

    def is_mature_male(self) -> bool:
        return self.sex == Sex.MALE and self.is_mature

    def is_mature_female(self) -> bool:
        return self.sex == Sex.FEMALE and self.is_mature

    def is_legal_male(self, params: IBMParameters) -> bool:
        """Size-based legal male check matching TethysBio::is_legal_male."""
        return (
            self.sex == Sex.MALE
            and self.carapace_width >= params.fishery.legal_size_male
            and self.carapace_width >= params.species.adult_size_threshold
        )

    def mass_if_mature_female(self) -> float:
        return self.mass if self.is_mature_female() else 0.0

    def mass_if_mature_male(self) -> float:
        return self.mass if self.is_mature_male() else 0.0

    def mass_if_legal_male(self, params: IBMParameters) -> float:
        return self.mass if self.is_legal_male(params) else 0.0


# ── Aggregate cohort & spatial exclusion ─────────────────────────────


@dataclass
class AggregateGroup:
    stage: LifeStage
    count: int
    mean_size: float
    age_days: float = 0.0
    degree_days_accumulated: float = 0.0
    id: int = 0
    stage_idx: int = 1
    abundance: float = 0.0
    mass: float = 0.0
    last_growth_time: float = 0.0
    pos: Tuple[int, int] = (50, 50)

    def __post_init__(self):
        if self.abundance == 0.0 and self.count > 0:
            self.abundance = float(self.count)
        elif self.count == 0 and self.abundance > 0.0:
            self.count = int(round(self.abundance))
        if self.stage_idx == 1 and self.stage != LifeStage.ZOEA:
            stage_map = {
                LifeStage.EGG: 0,
                LifeStage.ZOEA: 1,
                LifeStage.MEGALOPA: 3,
                LifeStage.JUVENILE: 4,
            }
            self.stage_idx = stage_map.get(self.stage, 1)


@dataclass
class ExclusionBox:
    id: int
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    active: bool = True


def is_in_exclusion_zones(pos: Tuple[float, float], boxes: List[ExclusionBox]) -> bool:
    """Return True if spatial coordinate `pos = (x, y)` lies inside any active ExclusionBox."""
    x, y = float(pos[0]), float(pos[1])
    for b in boxes:
        if b.active and b.xmin <= x <= b.xmax and b.ymin <= y <= b.ymax:
            return True
    return False


def fishery_active(params: IBMParameters) -> bool:
    """Return True when daily fishing probability is strictly positive."""
    return params.fishery.fishing_probability_daily > 0.0


def fishery_open(params: IBMParameters, day_of_year: float) -> bool:
    """Year-wrap-aware fishery season check matching TethysBio::fishery_open."""
    if not fishery_active(params):
        return False
    s = params.fishery.fishing_season_start
    e = params.fishery.fishing_season_end
    if s <= e:
        return s <= day_of_year <= e
    return day_of_year >= s or day_of_year <= e


def export_scenario_toml(params: IBMParameters, path: str) -> str:
    """Write an IBMParameters instance to a TOML configuration file."""
    from dataclasses import asdict
    import pathlib

    def _fmt_val(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return repr(v)
        if isinstance(v, str):
            return f'"{v}"'
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(_fmt_val(x) for x in v) + "]"
        return f'"{v}"'

    sections = [
        ("toggles", params.toggles),
        ("simulation", params.simulation),
        ("environment", params.environment),
        ("organism", params.organism),
        ("species", params.species),
        ("phenology", params.phenology),
        ("mortality", params.mortality),
        ("fishery", params.fishery),
        ("leslie", params.leslie),
        ("criteria", params.criteria),
        ("output", params.output),
    ]
    lines = ["# AgentCrab / TethysBio Scenario Configuration\n"]
    for name, sec in sections:
        lines.append(f"[{name}]\n")
        for k, v in asdict(sec).items():
            lines.append(f"{k} = {_fmt_val(v)}\n")
        lines.append("\n")
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    return str(out)


# ── Configuration loading ─────────────────────────────────────────────


def _load_section(cls, raw_sec: dict, section_name: str, aliases: Optional[dict] = None):
    import warnings

    aliases = aliases or {}
    valid = set(cls.__dataclass_fields__.keys())
    kwargs = {}
    for k, v in raw_sec.items():
        target = aliases.get(k, k)
        if target in valid:
            kwargs[target] = v
        else:
            warnings.warn(f"Unknown configuration key [{section_name}].{k} ignored", stacklevel=3)
    return cls(**kwargs)


def load_config(path: Optional[str] = None, layer_default: bool = True) -> IBMParameters:
    """Load IBMParameters from a TOML file (optionally layered over default.toml), warning on unknown keys."""
    import tomllib

    files = resolve_configuration_files(path) if layer_default else ([str(path)] if path else [])
    raw: dict = {}
    for fp in files:
        with open(fp, "rb") as f:
            sec_data = tomllib.load(f)
        for sec, vals in sec_data.items():
            if isinstance(vals, dict) and isinstance(raw.get(sec), dict):
                raw[sec].update(vals)
            else:
                raw[sec] = vals

    params = IBMParameters()

    if "fishery" in raw:
        params.fishery = _load_section(
            FisheryParams,
            raw["fishery"],
            "fishery",
            {"legal_size_mm": "legal_size_male", "sublegal_retention": "no_take_proportion"},
        )
    if "environment" in raw:
        params.environment = _load_section(EnvironmentParams, raw["environment"], "environment")
    if "species" in raw:
        params.species = _load_section(
            SpeciesParams,
            raw["species"],
            "species",
            {
                "maturity_l50_male": "cw50_maturity",
                "maturity_l50_female": "cw50_maturity_female",
                "maturity_slope": "maturity_slope_male",
                "growth_beta0": "growth_beta0_male",
                "growth_beta1": "growth_slp_male",
                "growth_maturity_reduction": "growth_maturity_reduction_male",
            },
        )
    if "toggles" in raw:
        params.toggles = _load_section(
            ProcessToggles,
            raw["toggles"],
            "toggles",
            {
                "bioenergetics": "use_energetics",
                "thermal_stress": "use_thermal_stress",
                "cannibalism": "use_cannibalism",
                "adult_senescence": "use_senescence",
                "spermathecal_storage": "use_spermatheca",
                "directed_habitat_taxis": "use_directed_movement",
                "degree_day_ecdysial_gating": "use_degree_days",
                "molting": "use_molting",
                "reproduction": "use_reproduction",
            },
        )
    if "phenology" in raw:
        params.phenology = _load_section(PhenologyParams, raw["phenology"], "phenology")
    if "mortality" in raw:
        params.mortality = _load_section(MortalityParams, raw["mortality"], "mortality")
    if "simulation" in raw:
        params.simulation = _load_section(SimulationParams, raw["simulation"], "simulation")
    if "leslie" in raw:
        params.leslie = _load_section(LeslieParams, raw["leslie"], "leslie")
    if "criteria" in raw:
        params.criteria = _load_section(CriteriaParams, raw["criteria"], "criteria")
    if "output" in raw:
        params.output = _load_section(OutputParams, raw["output"], "output")
    if "organism" in raw:
        params.organism = _load_section(OrganismParams, raw["organism"], "organism")

    # Sync EnvironmentParams.grid_size with SimulationParams.(grid_nx, grid_ny)
    if "simulation" in raw and ("grid_nx" in raw["simulation"] or "grid_ny" in raw["simulation"]):
        params.environment.grid_size = (params.simulation.grid_nx, params.simulation.grid_ny)
    elif "environment" in raw and "grid_size" in raw["environment"]:
        gx, gy = params.environment.grid_size
        params.simulation.grid_nx = int(gx)
        params.simulation.grid_ny = int(gy)

    return params


@dataclass
class NamedPredicate:
    name: str
    fn: object

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def named_predicate(name: str, fn) -> NamedPredicate:
    """Wrap a callable predicate with a display name."""
    return NamedPredicate(name=name, fn=fn)


@dataclass
class AggregatePopulations:
    """Container summarizing aggregate cohort groups across pelagic and early benthic stages."""

    groups: list = field(default_factory=list)

    def total_abundance(self) -> float:
        return float(sum(getattr(g, "abundance", g.count) for g in self.groups))


@dataclass
class LeslieMatrixFit:
    matrix: list = field(default_factory=list)
    r_squared: float = 0.0
    lambda_max: float = 1.0
    stage_labels: list = field(default_factory=list)
    is_fitted: bool = False


@dataclass
class MacroModelComparison:
    schaefer: dict = field(default_factory=dict)
    delay_difference: dict = field(default_factory=dict)
    leslie: dict = field(default_factory=dict)
    best_model: str = "schaefer"


@dataclass
class EmpiricalBiologicalModel:
    l50_male: float = 91.8
    l50_female: float = 51.6
    slope_male: float = 6.9957
    slope_female: float = 12.9812
    growth_int_male: float = 0.30153
    growth_slp_male: float = 1.00027
    growth_int_female: float = 0.1931
    growth_slp_female: float = 1.0372
    stage_proportions: list = field(default_factory=list)


def is_mature_male(org: Organism) -> bool:
    return org.sex == Sex.MALE and org.is_mature


def is_mature_female(org: Organism) -> bool:
    return org.sex == Sex.FEMALE and org.is_mature


def mass_if_mature_female(org: Organism) -> float:
    return org.mass if is_mature_female(org) else 0.0


def mass_if_mature_male(org: Organism) -> float:
    return org.mass if is_mature_male(org) else 0.0


def mass_if_legal_male(org: Organism, params: IBMParameters) -> float:
    return org.mass if org.is_legal_male(params) else 0.0


def prototype_organism(params: Optional[IBMParameters] = None) -> Organism:
    """Create a baseline prototype Organism matching TethysBio::prototype_organism."""
    return Organism(
        id=0,
        sex=Sex.MALE,
        stage=LifeStage.JUVENILE,
        carapace_width=50.0,
        molt_stage=11,
        position=(0, 0),
    )


def default_agent_observations() -> dict:
    """Return standard agent observation accessors matching TethysBio."""
    return {
        "mature_female": is_mature_female,
        "mature_male": is_mature_male,
        "mass_mature_female": mass_if_mature_female,
        "mass_mature_male": mass_if_mature_male,
    }


def resolve_configuration_files(config_path: Optional[str] = None) -> list:
    """Resolve layered TOML configuration files (default.toml + species/user config)."""
    from pathlib import Path

    files = []
    pkg_root = Path(__file__).resolve().parents[2]
    default_toml = pkg_root / "configs" / "default.toml"
    if default_toml.exists():
        files.append(str(default_toml))
    if config_path:
        p = Path(config_path)
        if p.exists() and str(p.resolve()) not in {str(Path(f).resolve()) for f in files}:
            files.append(str(p))
    return files


def with_overrides(params: IBMParameters, overrides: dict) -> IBMParameters:
    """Return a deep copy of params with dotted or flat overrides applied."""
    import copy

    updated = copy.deepcopy(params)
    for key, val in overrides.items():
        if "." in key:
            sec, field_name = key.split(".", 1)
            if hasattr(updated, sec) and hasattr(getattr(updated, sec), field_name):
                setattr(getattr(updated, sec), field_name, val)
        else:
            for sec in (
                "fishery",
                "environment",
                "species",
                "toggles",
                "phenology",
                "mortality",
                "simulation",
                "leslie",
                "criteria",
                "output",
                "organism",
            ):
                sub = getattr(updated, sec)
                if hasattr(sub, key):
                    setattr(sub, key, val)
                    break
    return updated


def reconstruct_params(params: IBMParameters, **kwargs) -> IBMParameters:
    """Reconstruct IBMParameters applying keyword overrides."""
    return with_overrides(params, kwargs)
