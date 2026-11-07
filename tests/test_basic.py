"""Basic smoke tests for AgentCrab."""

import pytest
import numpy as np
from pathlib import Path
from agentcrab.types import IBMParameters, Organism, LifeStage, Sex, load_config
from agentcrab.species import get_stage_size, predict_postgrowth_size, maturity_probability
from agentcrab.dynamics import calculate_mortality_probability, check_molt_conditions
from agentcrab.simulation import CrabModel
from agentcrab.analysis import (
    fit_schaefer, build_leslie_matrix, summarize_leslie_model,
    evaluate_population_criteria, mohn_rho, schaefer_surplus_production,
)
from agentcrab.aoi import environmental_input_status


def test_organism_creation():
    org = Organism(unique_id=1, sex=Sex.MALE, stage=LifeStage.JUVENILE, carapace_width=20.0)
    assert org.is_adult() is False
    assert org.mass > 0


def test_params_default():
    params = IBMParameters()
    assert params.fishery.fishing_probability_daily == 0.005
    assert params.environment.benthic_carrying_capacity == 10000.0
    assert params.leslie.survival_stage1 == 0.74
    assert params.criteria.min_population == 1
    assert params.output.save is True


def test_config_loading():
    params = load_config("configs/snowcrab.toml")
    assert params.species.max_size_male == 160.0
    assert params.species.max_size_female == 95.0
    assert params.mortality.base_mortality == 0.001


def test_environmental_data_uses_project_local_directory():
    status = environmental_input_status()
    expected = (Path(__file__).resolve().parents[1] / "data" / "environmental").resolve()
    assert Path(status["environmental_data_dir"]).resolve() == expected
    assert "lakehouse_root" not in status


def test_model_creation():
    model = CrabModel(n_agents=10, grid_size=(10, 10), seed=42)
    assert len(model.agents) == 10
    summary = model.summarize()
    assert summary["n_agents"] == 10


def test_model_step():
    model = CrabModel(n_agents=10, grid_size=(10, 10), seed=42)
    model.step()
    assert len(model.agents) <= 10


def test_model_run():
    model = CrabModel(n_agents=10, grid_size=(10, 10), seed=42)
    model_df, agent_df = model.run(5, when=1)
    assert len(model_df) > 0
    assert "n_agents" in model_df.columns


def test_mortality_probability():
    org = Organism(unique_id=1, sex=Sex.MALE, stage=LifeStage.ADULT, carapace_width=100.0)
    params = IBMParameters()
    prob = calculate_mortality_probability(org, params, 10.0, 100.0)
    assert 0.0 <= prob <= 1.0


def test_stage_size():
    params = IBMParameters()
    size_male = get_stage_size(LifeStage.ADULT, params, Sex.MALE)
    size_female = get_stage_size(LifeStage.ADULT, params, Sex.FEMALE)
    assert size_male > size_female  # males larger


def test_maturity_probability_sex_difference():
    params = IBMParameters()
    org_male = Organism(unique_id=1, sex=Sex.MALE, stage=LifeStage.ADULT, carapace_width=90.0)
    org_female = Organism(unique_id=2, sex=Sex.FEMALE, stage=LifeStage.ADULT, carapace_width=50.0)
    p_male = maturity_probability(org_male, params, 4.0)
    p_female = maturity_probability(org_female, params, 4.0)
    assert 0.0 <= p_male <= 1.0
    assert 0.0 <= p_female <= 1.0


def test_leslie_matrix():
    params = IBMParameters()
    L = build_leslie_matrix(params, 5)
    assert L.shape == (5, 5)
    s = summarize_leslie_model(L)
    assert "growth_rate_r" in s
    assert "stable_age_distribution" in s


def test_schaefer_fit():
    sp = np.array([100, 200, 300, 400, 500], dtype=float)
    rec = np.array([50, 80, 100, 120, 140], dtype=float)
    fit = fit_schaefer(sp, rec)
    assert "r" in fit
    assert "K" in fit
    assert fit["rmse"] >= 0


def test_mohn_rho():
    rho = mohn_rho(np.array([100, 110, 120, 130, 140], dtype=float))
    assert isinstance(rho, float)


def test_surplus_production():
    B = np.array([10.0, 20.0, 30.0])
    P = schaefer_surplus_production(B, 0.5, 100.0)
    assert len(P) == 3
    assert all(P >= 0)
