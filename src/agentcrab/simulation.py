"""Simulation engine for AgentCrab using Mesa 3.x."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from .types import (
    AggregateGroup,
    IBMParameters,
    LifeStage,
    Organism,
    Sex,
    load_config,
)
from .species import (
    default_species_behavior,
    calculate_mass,
    get_stage_size,
    predict_postgrowth_size,
)
from .dynamics import (
    calculate_energy_change,
    calculate_mortality_probability,
    perform_development,
    perform_reproduction,
    step_aggregate_groups,
    calculate_habitat_suitability,
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
    """Agent-based model for snow crab population dynamics."""

    def __init__(
        self,
        n_agents: int = 2000,
        grid_size: Tuple[int, int] = (1000, 1000),
        seed: Optional[int] = None,
        params: Optional[IBMParameters] = None,
        config_path: Optional[str] = None,
    ):
        super().__init__(seed=seed)
        self.random = random.Random(seed) if seed is not None else random.Random()

        if config_path is not None:
            self.params = load_config(config_path)
        elif params is not None:
            self.params = params
        else:
            self.params = IBMParameters()

        self.grid_size = grid_size
        self.grid = MultiGrid(*grid_size, torus=False)
        self.catch_total = 0
        self.catch_legal = 0

        self.species_behavior = default_species_behavior()

        # Environmental fields (simplified: uniform for now)
        self.temperature = self.params.species.growth_ref_temp
        self.depth = 100.0  # meters

        # Aggregate cohorts (pre-recruit pelagic)
        self.aggregate_groups: list[AggregateGroup] = []

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "n_agents": lambda m: len(m.agents),
                "mean_cw": lambda m: self._mean_carapace_width(),
                "n_mature": lambda m: sum(1 for a in m.agents if a.organism.is_mature),
                "n_legal": lambda m: sum(1 for a in m.agents if a.organism.is_legal_male(m.params)),
                "n_foraging": lambda m: sum(1 for a in m.agents if a.organism.state == "foraging"),
                "n_resting": lambda m: sum(1 for a in m.agents if a.organism.state == "resting"),
                "n_molting": lambda m: sum(1 for a in m.agents if a.organism.state == "molting"),
                "n_dead": lambda m: sum(1 for a in m.agents if a.organism.state == "dead"),
                "biomass_kg": lambda m: sum(a.organism.mass for a in m.agents if a.organism.state != "dead"),
                "mature_females": lambda m: sum(1 for a in m.agents if a.organism.is_adult_female() and a.organism.is_mature),
                "mature_males": lambda m: sum(1 for a in m.agents if a.organism.is_adult_male() and a.organism.is_mature),
                "juvenile_recruits": lambda m: sum(1 for a in m.agents if a.organism.stage == LifeStage.JUVENILE),
                "catch_total": lambda m: m.catch_total,
                "catch_legal": lambda m: m.catch_legal,
            },
            agent_reporters={
                "carapace_width": lambda a: a.organism.carapace_width,
                "stage": lambda a: a.organism.stage.value,
                "sex": lambda a: a.organism.sex.value,
                "age_days": lambda a: a.organism.age_days,
                "energy": lambda a: a.organism.energy_reserves,
                "state": lambda a: a.organism.state,
            },
        )

        # Initialize agents
        self._initialize_agents(n_agents)
        self.running = True

    def _initialize_agents(self, n_agents: int):
        """Create initial population."""
        for i in range(n_agents):
            sex = Sex.MALE if self.random.random() < 0.5 else Sex.FEMALE
            stage = LifeStage.JUVENILE if self.random.random() < 0.7 else LifeStage.ADULT
            cw = get_stage_size(stage, self.params)
            cw = max(1.0, cw + self.random.gauss(0, cw * 0.1))
            organism = Organism(
                unique_id=i,
                sex=sex,
                stage=stage,
                carapace_width=cw,
                age_days=self.random.uniform(0, 365),
                energy_reserves=self.random.uniform(0.5, 1.0),
                is_mature=stage == LifeStage.ADULT,
                state="foraging",
                mass=calculate_mass(cw, sex, stage == LifeStage.ADULT, self.params),
            )
            agent = CrabAgent(self, organism)
            x = self.random.randrange(self.grid_size[0])
            y = self.random.randrange(self.grid_size[1])
            self.grid.place_agent(agent, (x, y))

    def _mean_carapace_width(self) -> float:
        agents = list(self.agents)
        if not agents:
            return 0.0
        return sum(a.organism.carapace_width for a in agents) / len(agents)

    def agent_step(self, agent: CrabAgent):
        """Execute one step for a single agent."""
        org = agent.organism
        params = self.params

        temp = self.temperature
        depth = self.depth
        suitability = calculate_habitat_suitability(temp, depth, params)

        energy_change = calculate_energy_change(org, params, temp)
        org.energy_reserves = max(0.0, min(1.0, org.energy_reserves + energy_change))

        org = perform_development(org, params, temp)
        agent.organism = org

        density = len(self.agents) / (self.grid_size[0] * self.grid_size[1])
        mort_prob = calculate_mortality_probability(org, params, temp, density)
        if self.random.random() < mort_prob:
            org.state = "dead"
            # Mesa 3.x: agents are removed via the scheduler; mark as dead instead
            # of forcibly removing mid-iteration to avoid iterator corruption.
            return

        fishery = params.fishery
        days_per_year = int(params.simulation.days_per_year)
        day_of_year = (self.steps + 1) % days_per_year
        if (
            fishery.fishing_season_start <= day_of_year <= fishery.fishing_season_end
            and org.is_legal_male(params)
            and self.random.random() < fishery.fishing_probability_daily
        ):
            self.catch_total += 1
            self.catch_legal += 1
            org.state = "harvested"
            agent.remove()
            return

        if params.toggles.directed_habitat_taxis:
            neighbors = self.grid.get_neighborhood(agent.pos, moore=True, include_center=False)
            if neighbors:
                new_pos = self.random.choice(neighbors)
                self.grid.move_agent(agent, new_pos)

    def step(self):
        """Execute one model step."""
        self.datacollector.collect(self)
        for agent in list(self.agents):
            agent.step()

        self.aggregate_groups = step_aggregate_groups(
            self.aggregate_groups, self.params, self.temperature
        )

        # Reproduction: mature females produce new juveniles (seasonal: once per year)
        days_per_year = int(self.params.simulation.days_per_year)
        day_of_year = (self.steps + 1) % days_per_year
        mating_start = int(self.params.phenology.mating_season_start)
        if day_of_year == mating_start:
            new_agents = []
            for agent in list(self.agents):
                org = agent.organism
                if org.is_adult_female() and org.is_mature:
                    fec = perform_reproduction(org, self.params)
                    if fec > 0:
                        # Use a density-dependent recruit number rather than hardcoded /10000
                        n_new = max(1, int(fec / 5000))
                        for _ in range(n_new):
                            new_id = len(self.agents) + len(new_agents)
                            new_org = Organism(
                                unique_id=new_id,
                                sex=Sex.MALE if self.random.random() < 0.5 else Sex.FEMALE,
                                stage=LifeStage.JUVENILE,
                                carapace_width=get_stage_size(LifeStage.JUVENILE, self.params),
                                age_days=0.0,
                                energy_reserves=self.random.uniform(0.5, 1.0),
                                state="foraging",
                                mass=calculate_mass(
                                    get_stage_size(LifeStage.JUVENILE, self.params),
                                    Sex.MALE if self.random.random() < 0.5 else Sex.FEMALE,
                                    False,
                                    self.params,
                                ),
                            )
                            new_agent = CrabAgent(self, new_org)
                            new_agents.append(new_agent)
                            x = self.random.randrange(self.grid_size[0])
                            y = self.random.randrange(self.grid_size[1])
                            self.grid.place_agent(new_agent, (x, y))

            for na in new_agents:
                self.register_agent(na)

    def run(self, n_steps: int, when: int = 1):
        """Run simulation for n_steps, collecting data every `when` steps."""
        for i in range(n_steps):
            self.step()
            if (i + 1) % when == 0:
                self.datacollector.collect(self)
        return self.datacollector.get_model_vars_dataframe(), self.datacollector.get_agent_vars_dataframe()

    def summarize(self) -> dict:
        """Return population summary statistics."""
        agents = list(self.agents)
        if not agents:
            return {
                "n_agents": 0,
                "mean_cw": 0.0,
                "n_mature": 0,
                "n_legal": 0,
                "n_foraging": 0,
                "n_resting": 0,
                "n_molting": 0,
                "n_dead": 0,
                "n_female": 0,
                "n_male": 0,
                "biomass_kg": 0.0,
                "mature_females": 0,
                "mature_males": 0,
                "juvenile_recruits": 0,
                "catch_total": self.catch_total,
                "catch_legal": self.catch_legal,
            }
        return {
            "n_agents": len(agents),
            "mean_cw": sum(a.organism.carapace_width for a in agents) / len(agents),
            "biomass_kg": sum(a.organism.mass for a in agents if a.organism.state != "dead"),
            "mature_females": sum(1 for a in agents if a.organism.is_adult_female() and a.organism.is_mature),
            "mature_males": sum(1 for a in agents if a.organism.is_adult_male() and a.organism.is_mature),
            "juvenile_recruits": sum(1 for a in agents if a.organism.stage == LifeStage.JUVENILE),
            "catch_total": self.catch_total,
            "catch_legal": self.catch_legal,
            "n_mature": sum(1 for a in agents if a.organism.is_mature),
            "n_legal": sum(1 for a in agents if a.organism.is_legal_male(self.params)),
            "n_foraging": sum(1 for a in agents if a.organism.state == "foraging"),
            "n_resting": sum(1 for a in agents if a.organism.state == "resting"),
            "n_molting": sum(1 for a in agents if a.organism.state == "molting"),
            "n_dead": sum(1 for a in agents if a.organism.state == "dead"),
            "n_female": sum(1 for a in agents if a.organism.sex == Sex.FEMALE),
            "n_male": sum(1 for a in agents if a.organism.sex == Sex.MALE),
        }
