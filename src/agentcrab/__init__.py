"""AgentCrab: Python (Mesa) port of TethysBio."""

__version__ = "0.1.0"

from .types import (
    IBMParameters,
    Organism,
    LifeStage,
    Sex,
    load_config,
)
from .species import (
    default_species_behavior,
    get_stage_size,
    predict_postgrowth_size,
)
from .dynamics import (
    calculate_energy_change,
    calculate_mortality_probability,
    perform_development,
    perform_reproduction,
)
from .simulation import CrabModel
from .analysis import (
    fit_schaefer,
    fit_delay_difference,
    build_leslie_matrix,
    summarize_leslie_model,
    evaluate_population_criteria,
    mohn_rho,
    schaefer_surplus_production,
)

__all__ = [
    "IBMParameters",
    "Organism",
    "LifeStage",
    "Sex",
    "load_config",
    "default_species_behavior",
    "get_stage_size",
    "predict_postgrowth_size",
    "calculate_energy_change",
    "calculate_mortality_probability",
    "perform_development",
    "perform_reproduction",
    "CrabModel",
    "fit_schaefer",
    "fit_delay_difference",
    "build_leslie_matrix",
    "summarize_leslie_model",
    "evaluate_population_criteria",
    "mohn_rho",
    "schaefer_surplus_production",
]