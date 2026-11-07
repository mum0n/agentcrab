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
    fishing_probability_daily: float = 0.005
    legal_size_mm: float = 95.0
    sublegal_retention: bool = False
    fishing_season_start: float = 90.0
    fishing_season_end: float = 150.0
    no_take_proportion: float = 0.0


@dataclass
class EnvironmentParams:
    temp_offset: float = 0.0
    temp_base: float = 0.0
    temp_multiplier: float = 1.0
    temp_noise_sd: float = 0.5
    temp_base_degreedays: float = -1.5
    benthic_carrying_capacity: float = 10000.0
    benthic_carrying_capacity_exponent: float = 2.0
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
    grid_size: Tuple[int, int] = (1000, 1000)
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
    growth_beta1_male: float = 1.00027
    growth_beta1_female: float = 1.0372
    growth_maturity_reduction_male: float = -0.13564
    growth_maturity_reduction_female: float = -0.131
    growth_temp_coeff: float = -0.004
    growth_ref_temp: float = 4.0
    growth_noise_sd: float = 0.04
    asymptotic_size: float = 160.0
    stage_lower_bound_kappa: float = 0.235833
    stage_lower_bound_s0: float = 1.102422
    stage_lower_bound_iota0: float = 4.0
    fecundity_primiparous_intercept: float = -0.7229
    fecundity_primiparous_slope: float = 2.94
    fecundity_multiparous_intercept: float = 0.0158
    fecundity_multiparous_slope: float = 2.5811
    fecundity_senescence_threshold: float = 3.0
    fecundity_senescence_floor: float = 0.6
    fecundity_senescence_rate: float = 0.08
    primiparous_shell_age_max: float = 2.0
    fecundity_parity_size_split: float = 65.0
    fecundity_stmarie_exponent: float = 2.725
    fecundity_stmarie_coefficient: float = 0.7311
    # Stage size ranges
    max_stage_male: int = 19
    max_stage_female: int = 16
    juvenile_max_stage: int = 12
    adult_size_threshold: float = 68.0
    # Wet-mass allometry
    mass_alpha_male_immature: float = 0.0002665
    mass_beta_male_immature: float = 3.098
    mass_alpha_male_mature: float = 0.000183
    mass_beta_male_mature: float = 3.16
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
    # Stage labels
    stage_labels: list = field(default_factory=lambda: [
        "Egg", "z1", "z2", "m", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "11", "12", "13", "14", "15", "16"
    ])
    juvenile_max_stage: int = 12
    adult_size_threshold: float = 68.0
    recruitment_stage: int = 13


@dataclass
class ProcessToggles:
    bioenergetics: bool = True
    thermal_stress: bool = True
    cannibalism: bool = True
    adult_senescence: bool = True
    spermathecal_storage: bool = True
    directed_habitat_taxis: bool = True
    degree_day_ecdysial_gating: bool = True
    molting: bool = True
    reproduction: bool = True
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
    energy_reserves: float = 1.0
    molt_stage: int = 0
    degree_days_accumulated: float = 0.0
    is_mature: bool = False
    is_ovigerous: bool = False
    shell_age_days: float = 0.0
    is_soft_shell: bool = False
    spermatheca_stored: float = 0.0
    position: Tuple[int, int] = (0, 0)
    state: str = "foraging"
    mass: float = 0.0

    def __post_init__(self):
        """Compute mass from carapace width using sex-specific allometry when not explicitly set."""
        if self.mass == 0.0:
            from .species import calculate_mass
            from .types import IBMParameters

            default_params = IBMParameters()
            self.mass = calculate_mass(self.carapace_width, self.sex, self.is_mature, default_params)

    def is_adult(self) -> bool:
        return self.stage == LifeStage.ADULT

    def is_adult_male(self) -> bool:
        return self.is_adult() and self.sex == Sex.MALE

    def is_adult_female(self) -> bool:
        return self.is_adult() and self.sex == Sex.FEMALE

    def is_legal_male(self, params: IBMParameters) -> bool:
        return (
            self.is_adult_male()
            and self.is_mature
            and self.carapace_width >= params.fishery.legal_size_mm
        )



# ── Aggregate cohort ──────────────────────────────────────────────────


@dataclass
class AggregateGroup:
    stage: LifeStage
    count: int
    mean_size: float
    age_days: float = 0.0
    degree_days_accumulated: float = 0.0


# ── Configuration loading ─────────────────────────────────────────────


def load_config(path: str) -> IBMParameters:
    """Load IBMParameters from a TOML file."""
    import tomllib
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    params = IBMParameters()

    if "fishery" in raw:
        params.fishery = FisheryParams(**{k: v for k, v in raw["fishery"].items() if k in FisheryParams.__dataclass_fields__})
    if "environment" in raw:
        params.environment = EnvironmentParams(**{k: v for k, v in raw["environment"].items() if k in EnvironmentParams.__dataclass_fields__})
    if "species" in raw:
        params.species = SpeciesParams(**{k: v for k, v in raw["species"].items() if k in SpeciesParams.__dataclass_fields__})
    if "toggles" in raw:
        params.toggles = ProcessToggles(**{k: v for k, v in raw["toggles"].items() if k in ProcessToggles.__dataclass_fields__})
    if "phenology" in raw:
        params.phenology = PhenologyParams(**{k: v for k, v in raw["phenology"].items() if k in PhenologyParams.__dataclass_fields__})
    if "mortality" in raw:
        params.mortality = MortalityParams(**{k: v for k, v in raw["mortality"].items() if k in MortalityParams.__dataclass_fields__})
    if "simulation" in raw:
        params.simulation = SimulationParams(**{k: v for k, v in raw["simulation"].items() if k in SimulationParams.__dataclass_fields__})
    if "leslie" in raw:
        params.leslie = LeslieParams(**{k: v for k, v in raw["leslie"].items() if k in LeslieParams.__dataclass_fields__})
    if "criteria" in raw:
        params.criteria = CriteriaParams(**{k: v for k, v in raw["criteria"].items() if k in CriteriaParams.__dataclass_fields__})
    if "output" in raw:
        params.output = OutputParams(**{k: v for k, v in raw["output"].items() if k in OutputParams.__dataclass_fields__})
    if "organism" in raw:
        params.organism = OrganismParams(**{k: v for k, v in raw["organism"].items() if k in OrganismParams.__dataclass_fields__})

    return params
