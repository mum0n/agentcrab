"""Tests verifying exact mathematical and biological parity with TethysBio."""

import math
import pytest
import numpy as np
from agentcrab.types import ExclusionBox, IBMParameters, LifeStage, Organism, Sex, fishery_open, is_in_exclusion_zones
from agentcrab.species import calculate_mass, default_species_behavior, maturity_probability, predict_postgrowth_size
from agentcrab.dynamics import (
    benthic_carrying_capacity,
    calculate_habitat_suitability,
    calculate_mortality_probability,
    is_empirically_measured,
    molt_probability,
    mortality_components,
    perform_molt,
)
from agentcrab.analysis import fit_delay_difference, fit_generalized_leslie_model, fit_schaefer, fit_stock_recruitment
from agentcrab.realtime_dashboard import seasonal_climatology


def test_mass_allometry():
    params = IBMParameters()
    # Male mature 95mm: 0.0002665 * 95^3.098
    m_mat_g = calculate_mass(95.0, Sex.MALE, True, params)
    assert math.isclose(m_mat_g, 0.0002665 * (95.0 ** 3.098), rel_tol=1e-6)

    # Male immature 50mm: 0.000183 * 50^3.16
    m_imm_g = calculate_mass(50.0, Sex.MALE, False, params)
    assert math.isclose(m_imm_g, 0.000183 * (50.0 ** 3.16), rel_tol=1e-6)

    # Female mature 70mm (grams)
    f_mat_g = calculate_mass(70.0, Sex.FEMALE, True, params)
    assert math.isclose(f_mat_g, 0.00045 * (70.0 ** 2.92), rel_tol=1e-6)

    # Female immature 40mm (grams)
    f_imm_g = calculate_mass(40.0, Sex.FEMALE, False, params)
    assert math.isclose(f_imm_g, 0.000344 * (40.0 ** 2.98), rel_tol=1e-6)


def test_hiatt_postgrowth():
    params = IBMParameters()
    # Male 80mm
    next_m = predict_postgrowth_size(80.0, params, 4.0, Sex.MALE)
    assert 105.0 < next_m < 112.0

    # Female 60mm
    next_f = predict_postgrowth_size(60.0, params, 4.0, Sex.FEMALE)
    assert 75.0 < next_f < 85.0

    # Caps (160mm male, 95mm female)
    cap_m = predict_postgrowth_size(155.0, params, 4.0, Sex.MALE)
    cap_f = predict_postgrowth_size(90.0, params, 4.0, Sex.FEMALE)
    assert cap_m <= 160.0
    assert cap_f <= 95.0


def test_maturity_probability():
    params = IBMParameters()
    # 35mm size gate
    org_small = Organism(unique_id=1, sex=Sex.MALE, stage=LifeStage.JUVENILE, carapace_width=34.0)
    p_small = maturity_probability(org_small, params, 4.0)
    assert p_small == 0.0

    # Male L50 = 91.8 mm
    org_l50_m = Organism(unique_id=2, sex=Sex.MALE, stage=LifeStage.ADULT, carapace_width=91.8)
    p_m = maturity_probability(org_l50_m, params, 4.0)
    assert abs(p_m - 0.5) < 1e-3

    # Female L50 = 51.6 mm
    org_l50_f = Organism(unique_id=3, sex=Sex.FEMALE, stage=LifeStage.ADULT, carapace_width=51.6)
    p_f = maturity_probability(org_l50_f, params, 4.0)
    assert abs(p_f - 0.5) < 1e-3


def test_empirically_measured_and_density_feedback():
    params = IBMParameters()
    org_juv = Organism(unique_id=1, sex=Sex.MALE, stage=LifeStage.JUVENILE, carapace_width=25.0, molt_stage=8)
    org_meas = Organism(unique_id=2, sex=Sex.MALE, stage=LifeStage.JUVENILE, carapace_width=40.0, molt_stage=11)

    assert not is_empirically_measured(org_juv)
    assert is_empirically_measured(org_meas)

    k = benthic_carrying_capacity(params, grid_size=(1000, 1000), unit_meters=1.0)
    assert k == 6000.0

    # Under-capacity vs over-capacity
    mort_low = calculate_mortality_probability(org_meas, params, 4.0, 10.0, n_measured=3000.0, k_cap=k)
    mort_high = calculate_mortality_probability(org_meas, params, 4.0, 10.0, n_measured=12000.0, k_cap=k)
    assert mort_high > mort_low


def test_seasonal_climatology():
    days = list(range(0, 365, 10))
    temps, mean_t, amp = seasonal_climatology(days, depth=150.0, lat=45.0)
    assert len(temps) == len(days)
    assert 2.0 < mean_t < 4.0
    assert amp > 0.5


def test_base10_fecundity_and_terminal_molt():
    params = IBMParameters()
    fem = Organism(
        unique_id=10, sex=Sex.FEMALE, stage=LifeStage.ADULT,
        carapace_width=70.0, molt_stage=14, is_mature=True, shell_age_days=100.0,
    )
    eggs = default_species_behavior().fecundity(fem, params)
    expected = 10.0 ** (-0.7229 + 2.94 * math.log10(70.0))
    assert math.isclose(eggs, expected, rel_tol=1e-6)
    assert molt_probability(fem, params, temp=4.0) == 0.0

    imm_male = Organism(
        unique_id=11, sex=Sex.MALE, stage=LifeStage.JUVENILE,
        carapace_width=100.0, molt_stage=18, is_mature=False,
    )
    perform_molt(imm_male, params, temp=4.0)
    assert imm_male.molt_stage == 19
    assert imm_male.is_mature


def test_competing_risks_hsi_and_macro_fits():
    params = IBMParameters()
    org = Organism(unique_id=12, sex=Sex.MALE, stage=LifeStage.ADULT, carapace_width=98.0, molt_stage=15)
    hz = mortality_components(org, params, temp=4.0, density=1.0, dt=1.0)
    p1 = calculate_mortality_probability(org, params, temp=4.0, density=1.0, dt=1.0)
    assert math.isclose(p1, 1.0 - math.exp(-sum(hz.values())), rel_tol=1e-9)
    assert org.is_legal_male(params)  # size-based without requiring is_mature

    assert calculate_habitat_suitability(2.0, 140.0) > 0.9
    assert calculate_habitat_suitability(11.0, 140.0) == 0.0
    assert calculate_habitat_suitability(2.0, 15.0) == 0.0

    params.fishery.fishing_probability_daily = 0.05
    params.fishery.fishing_season_start = 300.0
    params.fishery.fishing_season_end = 60.0
    assert fishery_open(params, 320.0)
    assert fishery_open(params, 20.0)
    assert not fishery_open(params, 150.0)

    box = ExclusionBox(id=1, name="MPA", xmin=10, xmax=20, ymin=10, ymax=20, active=True)
    assert is_in_exclusion_zones((15, 15), [box])

    obs = np.array([100.0, 120.0, 138.0, 150.0, 158.0, 162.0])
    c = np.array([5.0, 6.0, 7.0, 8.0, 8.0, 8.0])
    sch = fit_schaefer(obs, c, dynamic=True)
    dd = fit_delay_difference(obs, c)
    sr = fit_stock_recruitment(obs[:-1], obs[1:])
    les = fit_generalized_leslie_model(np.vstack([obs * 0.4, obs * 0.6]), params)
    assert sch["R2"] > 0.5 and sch["MSY"] > 0.0
    assert dd["R2"] > 0.5 and dd["M"] > 0.0
    assert sr["beverton_holt"] is not None and sr["ricker"] is not None
    assert les["lambda1"] > 0.5


def test_extended_tethysbio_parity_apis(tmp_path):
    from agentcrab import (
        CrabModel,
        Sex,
        assimilate_environmental_inputs,
        compute_empirical_ogive,
        evaluate_population_criteria,
        extract_annual_production,
        gaussian_kde,
        growth_schedule,
        load_config,
        resolve_configuration_files,
        save_simulation_results,
        load_simulation_results,
        selectivity,
        shell_condition,
        with_overrides,
    )
    from agentcrab.assimilation import aoi_habitat_suitability as assim_hsi
    from agentcrab.aoi import aoi_habitat_suitability as aoi_hsi

    assert abs(assim_hsi(2.0, 140.0) - aoi_hsi(2.0, 140.0)) < 1e-12
    files = resolve_configuration_files("configs/snowcrab.toml")
    assert len(files) >= 1
    p = with_overrides(load_config("configs/snowcrab.toml"), {"temp_offset": 1.25})
    assert p.environment.temp_offset == 1.25

    sched = growth_schedule(Sex.MALE, p, n_stages=6)
    assert len(sched) == 6 and all(sched[i] < sched[i + 1] for i in range(5))
    assert 0.45 < selectivity(p.species.selectivity_s50, p) < 0.55

    m = CrabModel(n_agents=40, grid_size=(20, 20), seed=7, params=p)
    mdf, adf = m.run(10, when=2)
    assert len(mdf) == 5
    assert shell_condition(m.agents[0].organism, p) in (1, 2, 3, 4, 5)

    ann = extract_annual_production(mdf, days_per_year=2.0)
    assert len(ann) >= 2
    save_simulation_results(mdf, adf, output_dir=str(tmp_path), days_per_year=2.0)
    loaded = load_simulation_results(str(tmp_path))
    assert "annual_production" in loaded and len(loaded["annual_production"]) == len(ann)

    crit = evaluate_population_criteria(mdf, p, agent_df=adf)
    assert "maturity_rmse" in crit and "growth_rmse" in crit
    g, dens = gaussian_kde([40.0, 60.0, 80.0])
    assert len(g) == len(dens) and float(dens.max()) > 0.0
    og = compute_empirical_ogive([30.0, 50.0, 95.0, 110.0], [0, 0, 1, 1])
    assert len(og["sizes"]) >= 2

    res = assimilate_environmental_inputs(p, output_dir=str(tmp_path / "env"))
    assert (tmp_path / "env" / "manifest.toml").exists()
    assert "status" in res
