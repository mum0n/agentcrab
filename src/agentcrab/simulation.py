"""Simulation engine for AgentCrab using Mesa 3.x, aligned with TethysBio."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

from .dynamics import (
    benthic_carrying_capacity,
    calculate_habitat_suitability,
    calculate_mortality_probability,
    dominant_mortality_cause,
    empirical_measured_abundance,
    mortality_components,
    perform_development,
    perform_reproduction,
    step_aggregate_groups,
)
from .species import calculate_mass, default_species_behavior, get_stage_size
from .types import (
    AggregateGroup,
    ExclusionBox,
    IBMParameters,
    LifeStage,
    Organism,
    Sex,
    fishery_open,
    is_in_exclusion_zones,
    load_config,
)


class CrabAgent(Agent):
    """Mesa agent wrapping an Organism."""

    def __init__(self, model: "CrabModel", organism: Organism):
        super().__init__(model)
        self.organism = organism

    def step(self):
        """Delegate to model's agent step logic."""
        self.model.agent_step(self)


class CrabModel(Model):
    """Agent-based model for snow crab population dynamics matching TethysBio."""

    def __init__(
        self,
        n_agents: Optional[int] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
        params: Optional[IBMParameters] = None,
        config_path: Optional[str] = None,
        mpa: Optional[dict] = None,
        exclusion_boxes: Optional[List[ExclusionBox]] = None,
        stage_multipliers: Optional[dict] = None,
        cw_shift: float = 0.0,
    ):
        if config_path is not None:
            loaded_params = load_config(config_path)
        elif params is not None:
            loaded_params = params
        else:
            loaded_params = IBMParameters()

        eff_seed = seed if seed is not None else loaded_params.simulation.seed
        super().__init__(seed=eff_seed)
        self.random = random.Random(eff_seed)
        self.params = loaded_params

        if grid_size is None:
            grid_size = (self.params.simulation.grid_nx, self.params.simulation.grid_ny)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.grid = MultiGrid(*self.grid_size, torus=False)

        self.dt = float(self.params.simulation.dt)
        self.elapsed_days = 0.0
        self.catch_total = 0
        self.catch_legal = 0
        self.total_catch_biomass_kg = 0.0
        self.deaths_fishery = 0
        self.deaths_natural = 0
        self.deaths_by_cause: Dict[str, int] = {
            "base": 0,
            "predation": 0,
            "soft_shell": 0,
            "starvation": 0,
            "thermal": 0,
            "senescence": 0,
            "cannibalism": 0,
        }

        self.mpa = mpa
        if exclusion_boxes is not None:
            self.exclusion_boxes: List[ExclusionBox] = list(exclusion_boxes)
        elif mpa is not None and mpa.get("active", False):
            self.exclusion_boxes = [
                ExclusionBox(
                    id=1,
                    name="MPA-1",
                    xmin=float(mpa.get("xmin", 0)),
                    xmax=float(mpa.get("xmax", self.grid_size[0])),
                    ymin=float(mpa.get("ymin", 0)),
                    ymax=float(mpa.get("ymax", self.grid_size[1])),
                    active=bool(mpa.get("active", True)),
                )
            ]
        else:
            self.exclusion_boxes = []

        self.stage_multipliers = stage_multipliers or {
            "pelagic": 1.0,
            "juvenile": 1.0,
            "male_adult": 1.0,
            "female_adult": 1.0,
        }
        self.cw_shift = float(cw_shift)
        self.species_behavior = default_species_behavior()

        # 2D environmental fields
        self._init_environmental_fields()
        self.temperature = float(np.mean(self.temperature_field))
        self.depth = float(np.mean(self.depth_field))

        self.aggregate_groups: List[AggregateGroup] = []
        self._next_uid = 1
        self._n_measured_cache = 0
        self._k_cap_cache = benthic_carrying_capacity(
            self.params, self.grid_size, self.params.simulation.unit_meters
        )

        self.datacollector = DataCollector(
            model_reporters={
                "n_agents": lambda m: len(m.agents),
                "mean_cw": lambda m: m._mean_carapace_width(),
                "n_mature": lambda m: sum(1 for a in m.agents if a.organism.is_mature),
                "n_legal": lambda m: sum(1 for a in m.agents if a.organism.is_legal_male(m.params)),
                "n_foraging": lambda m: sum(1 for a in m.agents if a.organism.state == "foraging"),
                "n_resting": lambda m: sum(1 for a in m.agents if a.organism.state == "resting"),
                "n_molting": lambda m: sum(1 for a in m.agents if a.organism.state == "molting"),
                "n_dead": lambda m: m.deaths_natural,
                "biomass_kg": lambda m: sum(a.organism.mass for a in m.agents) / 1000.0,
                "mature_females": lambda m: sum(
                    1 for a in m.agents if a.organism.sex == Sex.FEMALE and a.organism.is_mature
                ),
                "mature_males": lambda m: sum(
                    1 for a in m.agents if a.organism.sex == Sex.MALE and a.organism.is_mature
                ),
                "juvenile_recruits": lambda m: sum(
                    1 for a in m.agents if not a.organism.is_mature
                ),
                "catch_total": lambda m: m.catch_total,
                "catch_legal": lambda m: m.catch_legal,
                "catch_biomass_kg": lambda m: m.total_catch_biomass_kg,
            },
            agent_reporters={
                "carapace_width": lambda a: a.organism.carapace_width,
                "stage": lambda a: a.organism.molt_stage,
                "sex": lambda a: a.organism.sex.value,
                "age_days": lambda a: a.organism.age_days,
                "energy": lambda a: a.organism.energy_reserves,
                "state": lambda a: a.organism.state,
            },
        )

        eff_n = n_agents if n_agents is not None else self.params.simulation.n_agents
        self._initialize_agents(int(eff_n))
        self._n_measured_cache = empirical_measured_abundance(self)
        self.running = True

    def _init_environmental_fields(self) -> None:
        nx, ny = self.grid_size
        xs = np.linspace(0.0, 1.0, nx)[:, None]
        ys = np.linspace(0.0, 1.0, ny)[None, :]
        # Synthetic shelf-to-slope bathymetry (40 m - 320 m)
        self.depth_field = 140.0 + 95.0 * np.sin(2.5 * xs) * np.cos(2.0 * ys) + 45.0 * (1.0 - ys)
        self.depth_field = np.clip(self.depth_field, 25.0, 380.0)
        self.base_temperature_field = 2.8 - 0.004 * (self.depth_field - 140.0) + self.params.environment.temp_offset

        if getattr(self.params.environment, "use_geodata", False):
            from .assimilation import read_bathymetry, read_woa23_temperature

            lon_bounds = self.params.environment.domain_lon
            lat_bounds = self.params.environment.domain_lat
            lons = np.linspace(float(lon_bounds[0]), float(lon_bounds[1]), nx)
            lats = np.linspace(float(lat_bounds[0]), float(lat_bounds[1]), ny)
            probe_z = read_bathymetry(float(lats[ny // 2]), float(lons[nx // 2]))
            if probe_z is not None:
                for ix, lon in enumerate(lons):
                    for iy, lat in enumerate(lats):
                        z_val = read_bathymetry(float(lat), float(lon))
                        if z_val is not None:
                            self.depth_field[ix, iy] = max(5.0, float(z_val))
                        t_val = read_woa23_temperature(float(lat), float(lon), float(self.depth_field[ix, iy]))
                        if t_val is not None:
                            self.base_temperature_field[ix, iy] = float(t_val) + self.params.environment.temp_offset

        self.temperature_field = self.base_temperature_field.copy()
        self.surface_temperature_field = self.temperature_field + 4.0
        self.food_availability_field = np.full((nx, ny), float(self.params.environment.food_max) * 0.5)
        self.habitat_quality_field = np.zeros((nx, ny), dtype=float)
        self._refresh_habitat_quality()

    def _refresh_habitat_quality(self) -> None:
        nx, ny = self.grid_size
        # Vectorized HSI matching calculate_habitat_suitability
        t = self.temperature_field
        z = self.depth_field
        s_t = np.where(t >= 10.0, 0.0, 1.0 / (1.0 + np.exp(1.2 * (t - 5.0))))
        taper_t = np.where(t > 5.0, 0.5 * (1.0 + np.cos(math.pi * (t - 5.0) / 5.0)), 1.0)
        s_t = np.where(t >= 10.0, 0.0, s_t * taper_t)

        s_z = np.where((z <= 20.0) | (z >= 400.0), 0.0, np.exp(-0.5 * ((z - 140.0) / 90.0) ** 2))
        s_z = np.where(z < 50.0, s_z * 0.5 * (1.0 - np.cos(math.pi * (z - 20.0) / 30.0)), s_z)
        s_z = np.where(z > 300.0, s_z * 0.5 * (1.0 + np.cos(math.pi * (z - 300.0) / 100.0)), s_z)
        s_z = np.where((z <= 20.0) | (z >= 400.0), 0.0, s_z)

        hsi = self.params.environment.habitat_offset + self.params.environment.habitat_multiplier * (s_t * s_z)
        self.habitat_quality_field = np.clip(hsi, 0.0, 1.0)

    def _sample_marine_pos(self) -> Tuple[int, int]:
        nx, ny = self.grid_size
        for _ in range(16):
            x = self.random.randrange(nx)
            y = self.random.randrange(ny)
            if self.depth_field[x, y] >= 20.0 and self.habitat_quality_field[x, y] > 0.0:
                return (x, y)
        return (self.random.randrange(nx), self.random.randrange(ny))

    def _initialize_agents(self, n_agents: int) -> None:
        """Create initial population with stage weighting, optional aggregate cohorts, and trait heterogeneity."""
        agg_vec = getattr(self.params.simulation, "initial_aggregate_cohort", [])
        if agg_vec:
            juv_max = self.params.species.juvenile_max_stage
            for s_idx, cnt in enumerate(agg_vec):
                if s_idx < juv_max and float(cnt) > 0.0:
                    st_enum = (
                        LifeStage.EGG
                        if s_idx == 0
                        else (LifeStage.ZOEA if s_idx <= 2 else (LifeStage.MEGALOPA if s_idx == 3 else LifeStage.JUVENILE))
                    )
                    self.aggregate_groups.append(
                        AggregateGroup(
                            stage=st_enum,
                            count=int(round(cnt)),
                            mean_size=get_stage_size(s_idx, self.params, Sex.MALE, mode="mid"),
                            stage_idx=s_idx,
                            abundance=float(cnt),
                            pos=self._sample_marine_pos(),
                        )
                    )

        juv_w = max(0.05, self.stage_multipliers.get("juvenile", 1.0)) * 0.65
        ad_w = max(0.05, 0.5 * (self.stage_multipliers.get("male_adult", 1.0) + self.stage_multipliers.get("female_adult", 1.0))) * 0.35
        juv_prob = juv_w / (juv_w + ad_w)
        qmin = self.params.organism.trait_quantile_min
        qmax = self.params.organism.trait_quantile_max

        for _ in range(n_agents):
            sex = Sex.MALE if self.random.random() < 0.5 else Sex.FEMALE
            is_juv = self.random.random() < juv_prob
            max_st = self.params.species.max_stage_female if sex == Sex.FEMALE else self.params.species.max_stage_male
            if is_juv:
                molt_st = self.random.randint(8, self.params.species.juvenile_max_stage - 1)
                stage_enum = LifeStage.JUVENILE
                is_mat = False
            else:
                molt_st = self.random.randint(self.params.species.juvenile_max_stage, max_st)
                stage_enum = LifeStage.ADULT
                is_mat = True

            cw = get_stage_size(molt_st, self.params, sex=sex, mode="random", rng=self.random)
            if self.cw_shift != 0.0:
                cw = max(2.5, cw + self.cw_shift)

            org = Organism(
                unique_id=self._next_uid,
                sex=sex,
                stage=stage_enum,
                carapace_width=cw,
                age_days=molt_st * self.params.simulation.initial_age_days_per_stage,
                energy_reserves=self.params.organism.initial_energy,
                molt_stage=molt_st,
                degree_days_accumulated=self.random.uniform(
                    self.params.simulation.initial_degree_days_min,
                    self.params.simulation.initial_degree_days_max,
                ),
                is_mature=is_mat,
                is_mating_ready=(is_mat and sex == Sex.FEMALE),
                trait_growth_quantile=self.random.uniform(qmin, qmax),
                trait_maturity_quantile=self.random.uniform(qmin, qmax),
                trait_fecundity_quantile=self.random.uniform(qmin, qmax),
                trait_mortality_quantile=self.random.uniform(qmin, qmax),
                state="foraging",
            )
            org.mass = calculate_mass(cw, sex, is_mat, self.params, rng=self.random)
            self._next_uid += 1
            agent = CrabAgent(self, org)
            pos = self._sample_marine_pos()
            org.position = pos
            self.grid.place_agent(agent, pos)

    def _mean_carapace_width(self) -> float:
        agents = list(self.agents)
        if not agents:
            return 0.0
        return sum(a.organism.carapace_width for a in agents) / len(agents)

    def _disperse_agent(self, agent: CrabAgent) -> None:
        """Softmax habitat-taxis movement with land and thermal avoidance."""
        if not self.params.toggles.use_movement or agent.pos is None:
            return
        if agent.organism.state == "resting":
            return

        neighbors = self.grid.get_neighborhood(agent.pos, moore=True, include_center=True)
        if not neighbors:
            return

        if not self.params.toggles.use_directed_movement:
            new_pos = self.random.choice(neighbors)
            self.grid.move_agent(agent, new_pos)
            agent.organism.position = new_pos
            return

        env = self.params.environment
        t_opt = env.habitat_temperature_optimum
        t_sig = max(0.1, env.habitat_temperature_sigma)
        f_w = env.habitat_food_weight
        beta = 1.0 / max(1e-3, env.movement_softmax_temperature)

        valid_cells = []
        weights = []
        for cx, cy in neighbors:
            d = float(self.depth_field[cx, cy])
            t = float(self.temperature_field[cx, cy])
            hsi = float(self.habitat_quality_field[cx, cy])
            if d < 20.0 or hsi <= 0.0 or t >= 10.0:
                continue
            s_temp = math.exp(-0.5 * ((t - t_opt) / t_sig) ** 2)
            food = float(self.food_availability_field[cx, cy])
            s_food = food / (food + max(1e-6, env.half_saturation_food))
            util = hsi * s_temp * (1.0 + f_w * s_food)
            valid_cells.append((cx, cy))
            weights.append(math.exp(min(30.0, beta * util)))

        if not valid_cells:
            return
        total_w = sum(weights)
        if total_w <= 0.0:
            new_pos = self.random.choice(valid_cells)
        else:
            r = self.random.random() * total_w
            acc = 0.0
            new_pos = valid_cells[-1]
            for pos, w in zip(valid_cells, weights):
                acc += w
                if acc >= r:
                    new_pos = pos
                    break
        self.grid.move_agent(agent, new_pos)
        agent.organism.position = new_pos

    def agent_step(self, agent: CrabAgent) -> None:
        """Execute one dt-scaled step for a single agent."""
        org = agent.organism
        params = self.params
        dt = self.dt
        dpy = params.simulation.days_per_year
        doy = self.elapsed_days % dpy

        px, py = agent.pos if agent.pos is not None else (0, 0)
        temp = float(self.temperature_field[px, py])
        food_avail = float(self.food_availability_field[px, py])

        org = perform_development(
            org,
            params,
            temp,
            dt=dt,
            day_of_year=doy,
            current_time=self.elapsed_days,
            food_avail=food_avail,
            rng=self.random,
        )
        agent.organism = org

        # Competing-risks natural mortality
        area_cells = max(1.0, float(self.grid_size[0] * self.grid_size[1]))
        density = self._n_measured_cache / area_cells
        hazards = mortality_components(
            org,
            params,
            temp,
            density,
            dt=dt,
            n_measured=self._n_measured_cache,
            k_cap=self._k_cap_cache,
        )
        h_total = sum(hazards.values())
        mort_prob = min(params.mortality.hazard_cap, 1.0 - math.exp(-h_total)) if h_total > 0.0 else 0.0
        if self.random.random() < mort_prob:
            cause = dominant_mortality_cause(hazards)
            self.deaths_natural += 1
            self.deaths_by_cause[cause] = self.deaths_by_cause.get(cause, 0) + 1
            org.state = "dead"
            agent.remove()
            return

        # dt-scaled, selectivity-weighted fishery harvest
        if fishery_open(params, doy) and org.is_legal_male(params) and agent.pos is not None:
            gx, gy = agent.pos
            no_take_cols = int(math.floor(self.grid_size[0] * params.fishery.no_take_proportion))
            in_no_take = gx < no_take_cols
            in_mpa = is_in_exclusion_zones(agent.pos, self.exclusion_boxes)
            if not in_no_take and not in_mpa:
                sp = params.species
                sel = 1.0 / (
                    1.0 + math.exp(-(org.carapace_width - sp.selectivity_s50) / max(0.1, sp.selectivity_s50_sigma))
                )
                p_daily = max(0.0, min(0.999999, params.fishery.fishing_probability_daily * sel))
                p_harvest = -math.expm1(dt * math.log1p(-p_daily)) if p_daily > 0.0 else 0.0
                if self.random.random() < p_harvest:
                    self.catch_total += 1
                    self.catch_legal += 1
                    self.deaths_fishery += 1
                    self.total_catch_biomass_kg += org.mass / 1000.0
                    org.state = "harvested"
                    agent.remove()
                    return

        self._disperse_agent(agent)

    def _step_reproduction(self, doy: float) -> None:
        params = self.params
        if not params.toggles.use_reproduction:
            return
        ph = params.phenology
        in_mating = (
            ph.mating_season_start <= doy <= ph.mating_season_end
            if ph.mating_season_start <= ph.mating_season_end
            else (doy >= ph.mating_season_start or doy <= ph.mating_season_end)
        )

        # Release hatched broods into pelagic AggregateGroup stage 1
        for agent in list(self.agents):
            org = agent.organism
            if org.is_ovigerous and org.hatch_day >= 0.0 and self.elapsed_days >= org.hatch_day:
                if org.egg_count > 0.0:
                    pos = agent.pos if agent.pos is not None else (0, 0)
                    self.aggregate_groups.append(
                        AggregateGroup(
                            stage=LifeStage.ZOEA,
                            count=int(round(org.egg_count)),
                            mean_size=get_stage_size(1, params, Sex.MALE, mode="mid"),
                            stage_idx=1,
                            abundance=float(org.egg_count),
                            pos=pos,
                        )
                    )
                org.is_ovigerous = False
                org.egg_count = 0.0
                org.hatch_day = -1.0
                org.is_mating_ready = True

        if not in_mating:
            return

        mature_males = [a for a in self.agents if a.organism.sex == Sex.MALE and a.organism.is_mature]
        for agent in list(self.agents):
            org = agent.organism
            if org.sex != Sex.FEMALE or not org.is_mature or org.is_ovigerous or not org.is_mating_ready:
                continue

            mated = False
            if params.toggles.use_spermatheca and org.sperm_clutches > 0:
                org.sperm_clutches -= 1
                mated = True
            elif mature_males:
                mate = self.random.choice(mature_males)
                if mate.organism.energy_reserves >= ph.mating_energy_cost:
                    mate.organism.energy_reserves -= ph.mating_energy_cost
                    if params.toggles.use_spermatheca:
                        clutches = (
                            ph.sperm_stores_legal
                            if mate.organism.is_legal_male(params)
                            else ph.sperm_stores_sublegal
                        )
                        org.sperm_clutches = max(0, clutches - 1)
                    mated = True

            if mated:
                fec = perform_reproduction(org, params, rng=self.random)
                if fec > 0.0:
                    px, py = agent.pos if agent.pos is not None else (0, 0)
                    t_loc = max(ph.brood_temperature_floor, float(self.temperature_field[px, py]))
                    brood_days = (
                        ph.brood_duration_base
                        * (ph.brood_temperature_reference / t_loc) ** ph.brood_temperature_exponent
                        + self.random.gauss(0.0, ph.brood_duration_sd)
                    )
                    org.is_ovigerous = True
                    org.is_mating_ready = False
                    org.egg_count = fec
                    org.hatch_day = self.elapsed_days + max(60.0, brood_days)

    def step(self, collect: bool = True) -> None:
        """Execute one model step of duration dt days."""
        if collect:
            self.datacollector.collect(self)
        params = self.params
        dt = self.dt
        dpy = params.simulation.days_per_year
        doy = self.elapsed_days % dpy

        # Seasonal temperature cycling and food replenishment
        season_arg = 2.0 * math.pi * (doy - params.simulation.temp_seasonal_phase_day) / dpy
        self.temperature_field = (
            self.base_temperature_field + params.environment.temp_multiplier * math.sin(season_arg)
        )
        f_max = params.environment.food_max
        f_rate = params.environment.food_replenishment_rate
        self.food_availability_field = f_max + (self.food_availability_field - f_max) * math.exp(-f_rate * dt)
        self._refresh_habitat_quality()
        self.temperature = float(np.mean(self.temperature_field))

        self._n_measured_cache = empirical_measured_abundance(self)
        self._k_cap_cache = benthic_carrying_capacity(
            params, self.grid_size, params.simulation.unit_meters
        )

        for agent in list(self.agents):
            agent.step()

        self._step_reproduction(doy)

        # Advance aggregate cohorts and settle juveniles
        self.aggregate_groups, settled = step_aggregate_groups(
            self.aggregate_groups, params, self.temperature, dt=dt, rng=self.random
        )
        for n_rec, _, pos in settled:
            for _ in range(int(n_rec)):
                sex = Sex.MALE if self.random.random() < 0.5 else Sex.FEMALE
                cw = get_stage_size(params.species.juvenile_max_stage, params, sex=sex, mode="random", rng=self.random)
                org = Organism(
                    unique_id=self._next_uid,
                    sex=sex,
                    stage=LifeStage.JUVENILE,
                    carapace_width=cw,
                    energy_reserves=params.organism.initial_energy,
                    molt_stage=params.species.juvenile_max_stage,
                    state="foraging",
                )
                org.mass = calculate_mass(cw, sex, False, params, rng=self.random)
                self._next_uid += 1
                na = CrabAgent(self, org)
                s_pos = self._sample_marine_pos() if pos is None else pos
                org.position = s_pos
                self.grid.place_agent(na, s_pos)

        self.elapsed_days += dt

    def run(self, n_steps: int, when: int = 1):
        """Run simulation for n_steps, collecting data every `when` steps."""
        cadence = max(1, int(when))
        for i in range(n_steps):
            self.step(collect=((i + 1) % cadence == 0))
        return self.datacollector.get_model_vars_dataframe(), self.datacollector.get_agent_vars_dataframe()

    def summarize(self) -> dict:
        """Return population summary statistics."""
        agents = list(self.agents)
        if not agents:
            return {"n_agents": 0, "mean_cw": 0.0, "biomass_kg": 0.0, "n_mature": 0, "n_legal": 0}
        return {
            "n_agents": len(agents),
            "mean_cw": sum(a.organism.carapace_width for a in agents) / len(agents),
            "biomass_kg": sum(a.organism.mass for a in agents) / 1000.0,
            "n_mature": sum(1 for a in agents if a.organism.is_mature),
            "n_legal": sum(1 for a in agents if a.organism.is_legal_male(self.params)),
            "catch_total": self.catch_total,
            "catch_legal": self.catch_legal,
            "catch_biomass_kg": self.total_catch_biomass_kg,
        }
