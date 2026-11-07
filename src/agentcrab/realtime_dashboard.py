"""Real-time Solara dashboard for AgentCrab — matching the HTML/GLMakie reference
dashboard (agentcrabr/dashboard.html) panel-for-panel.

Panels implemented:
  0. Area of Interest (AOI) Map & Location Picker
  1. Spatial Habitat & Organism Distribution
  2. State Distribution (Current)
  3. Environmental Driver Distributions
  4. Abundance Trajectories
  5. Biomass, Catch & Macro-Models
  6. Living Stage Structure
  7. Stock-Recruitment Relationship
  8. Size Frequency Spectra by Sex & Maturity
  9. Maturity Ogives & Selectivity
  10. Multi-Attribute Calibration Criteria

Plus KPI strip and raw data table, matching the reference HTML dashboard.
"""

from __future__ import annotations

import solara
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from agentcrab.simulation import CrabModel
from agentcrab.aoi import (
    aoi_bottom_temperature_at,
    aoi_habitat_suitability,
    environmental_input_status,
    read_bathymetry,
)

PRESET_LOCATIONS = {
    "Scotian Shelf (central)": (-60.0, 45.0, 150.0),
    "Gulf of St. Lawrence": (-62.0, 48.5, 80.0),
    "Bay of Fundy": (-66.0, 44.5, 60.0),
    "Grand Banks": (-52.0, 46.0, 200.0),
    "Georges Bank": (-67.5, 41.5, 100.0),
}


def _viewport_environment(lons, lats):
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    depths = np.empty_like(lon_grid)
    temperatures = np.empty_like(lon_grid)
    fallback = 90 + 28 * np.abs(lat_grid - 45.0) + 10 * np.abs(lon_grid + 60.0)
    for index, (lon, lat) in enumerate(zip(lon_grid.ravel(), lat_grid.ravel())):
        depth = read_bathymetry(float(lat), float(lon))
        if depth is None:
            depth = float(fallback.ravel()[index])
        depths.ravel()[index] = depth
        temperatures.ravel()[index] = aoi_bottom_temperature_at(float(lon), float(lat), depth)
    suitability = np.vectorize(aoi_habitat_suitability)(temperatures, depths)
    return lon_grid, lat_grid, depths, temperatures, suitability


@solara.component
def RealtimeDashboard():
    # ---- Reactive state ----
    n_agents, set_n_agents = solara.use_state(200)
    grid_size, set_grid_size = solara.use_state(50)
    seed, set_seed = solara.use_state(42)
    step_count, set_step_count = solara.use_state(0)
    max_steps, set_max_steps = solara.use_state(365)

    # AOI location state
    selected_lon, set_selected_lon = solara.use_state(-60.0)
    selected_lat, set_selected_lat = solara.use_state(45.0)
    selected_depth, set_selected_depth = solara.use_state(150.0)
    selected_preset, set_selected_preset = solara.use_state("Scotian Shelf (central)")

    # Model and history
    model_state, set_model_state = solara.use_state(None)
    history, set_history = solara.use_state([])

    # ---- Derived environmental values ----
    bottom_temp = aoi_bottom_temperature_at(selected_lon, selected_lat, selected_depth)
    hsi = aoi_habitat_suitability(bottom_temp, selected_depth)

    # ---- Actions ----
    def init_model():
        m = CrabModel(
            n_agents=n_agents,
            grid_size=(grid_size, grid_size),
            seed=seed,
        )
        m.temperature = bottom_temp
        m.depth = selected_depth
        set_model_state(m)
        initial = m.summarize()
        initial["step"] = 0
        set_history([initial])
        set_step_count(0)

    def apply_preset(name: str):
        if name == "Custom":
            return
        lon, lat, depth = PRESET_LOCATIONS[name]
        set_selected_lon(lon)
        set_selected_lat(lat)
        set_selected_depth(depth)
        set_selected_preset(name)
        if model_state is not None:
            model_state.temperature = aoi_bottom_temperature_at(lon, lat, depth)
            model_state.depth = depth
            set_model_state(model_state)

    def update_aoi(lon=None, lat=None, depth=None):
        next_lon = selected_lon if lon is None else float(lon)
        next_lat = selected_lat if lat is None else float(lat)
        next_depth = selected_depth if depth is None else float(depth)
        set_selected_lon(next_lon)
        set_selected_lat(next_lat)
        set_selected_depth(next_depth)
        set_selected_preset("Custom")
        if model_state is not None:
            model_state.temperature = aoi_bottom_temperature_at(next_lon, next_lat, next_depth)
            model_state.depth = next_depth
            set_model_state(model_state)

    def step_once():
        m = model_state
        if m is not None and step_count < max_steps:
            m.step()
            summary = m.summarize()
            summary["step"] = m.steps
            set_step_count(m.steps)
            set_history(history + [summary])
            set_model_state(m)

    def run_steps(n: int):
        m = model_state
        if m is None:
            return
        next_history = list(history)
        completed_steps = step_count
        for _ in range(n):
            if completed_steps >= max_steps:
                break
            m.step()
            completed_steps += 1
            summary = m.summarize()
            summary["step"] = m.steps
            next_history.append(summary)
        set_step_count(completed_steps)
        set_history(next_history)
        set_model_state(m)

    # Auto-init
    if model_state is None:
        init_model()

    m = model_state
    summary = m.summarize() if m is not None else None

    with solara.Card("AgentCrab Dashboard — TethysBio Reference Layout"):
        # ---- Controls row ----
        with solara.Row():
            solara.InputInt("Agents", value=n_agents, on_value=set_n_agents)
            solara.InputInt("Grid", value=grid_size, on_value=set_grid_size)
            solara.InputInt("Seed", value=seed, on_value=set_seed)
            solara.InputInt("Max Steps", value=max_steps, on_value=set_max_steps)

        with solara.Row():
            solara.Button("Init", on_click=init_model, color="primary")
            solara.Button("Step", on_click=step_once, color="secondary")
            solara.Button("Run 10", on_click=lambda: run_steps(10), color="success")
            solara.Button("Run 50", on_click=lambda: run_steps(50), color="info")

        # ---- KPI strip (matching HTML reference) ----
        if m is not None and summary is not None:
            with solara.Row():
                solara.Text(f"Total Crabs: {summary['n_agents']}")
                solara.Text(f"Foraging: {summary.get('n_foraging', summary['n_agents'])}")
                solara.Text(f"Dead: {summary.get('n_dead', 0)}")
                solara.Text(f"Mean Size: {summary['mean_cw']:.2f} mm")
                solara.Text(f"Legal catch: {summary.get('catch_legal', 0)}")

        # ---- Panel 0: AOI Map & Location Picker ----
        with solara.Card("0. Area of Interest (AOI) Map & Location Picker"):
            lon_grid, lat_grid, depths, _, _ = _viewport_environment(
                np.linspace(-66.5, -57.5, 24), np.linspace(42.0, 48.5, 20)
            )
            bathymetry_available = environmental_input_status()["nwa_bathymetry"]
            map_figure = go.Figure()
            map_figure.add_trace(go.Scattergeo(
                lon=lon_grid.ravel(),
                lat=lat_grid.ravel(),
                mode="markers",
                marker={
                    "size": 6,
                    "color": depths.ravel(),
                    "colorscale": "Blues",
                    "reversescale": True,
                    "colorbar": {"title": "Depth (m)"},
                    "opacity": 0.7,
                },
                name="Bathymetry",
                hovertemplate="Lon %{lon:.2f}°<br>Lat %{lat:.2f}°<br>Depth %{marker.color:.0f} m<extra></extra>",
            ))
            map_figure.add_trace(go.Scattergeo(
                lon=[selected_lon],
                lat=[selected_lat],
                mode="markers",
                marker={"size": 14, "symbol": "star", "color": "red"},
                name="Selected AOI",
            ))
            map_figure.update_layout(
                title=(
                    "Area of Interest (AOI) Selection"
                    if bathymetry_available
                    else "Area of Interest (AOI) Selection — illustrative bathymetry"
                ),
                geo={
                    "scope": "north america",
                    "showcoastlines": True,
                    "coastlinecolor": "#475569",
                    "showland": True,
                    "landcolor": "#cbd5e1",
                    "showocean": True,
                    "oceancolor": "#e0f2fe",
                    "lonaxis": {"range": [-66.5, -57.5]},
                    "lataxis": {"range": [42.0, 48.5]},
                },
                margin={"l": 8, "r": 8, "t": 44, "b": 8},
                height=420,
            )

            def select_aoi(click_data):
                points = click_data.get("points", [])
                if not points:
                    return
                point = points[0]
                lon = point.get("lon", point.get("x"))
                lat = point.get("lat", point.get("y"))
                if lon is None or lat is None:
                    return
                update_aoi(lon=lon, lat=lat)

            solara.FigurePlotly(map_figure, on_click=select_aoi)

            with solara.Row():
                solara.Select(
                    "Preset", value=selected_preset,
                    values=["Custom", *PRESET_LOCATIONS.keys()], on_value=apply_preset
                )
                solara.InputFloat("Longitude", value=selected_lon, on_value=lambda value: update_aoi(lon=value))
                solara.InputFloat("Latitude", value=selected_lat, on_value=lambda value: update_aoi(lat=value))
                solara.InputFloat("Depth (m)", value=selected_depth, on_value=lambda value: update_aoi(depth=value))

            with solara.Row():
                solara.Text(f"Bottom Temp: {bottom_temp:.2f} °C")
                solara.Text(f"HSI: {hsi:.3f}")

        # ---- Panel 1: Spatial Habitat & Organism Distribution ----
        with solara.Card("1. Spatial Habitat & Organism Distribution"):
            if m is not None:
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                positions = [(a.pos[0], a.pos[1]) for a in m.agents]
                if positions:
                    xs, ys = zip(*positions)
                    ax1.scatter(xs, ys, c="blue", s=5, alpha=0.5, edgecolors="black", linewidths=0.3)
                ax1.set_xlim(0, m.grid_size[0])
                ax1.set_ylim(0, m.grid_size[1])
                ax1.set_xlabel("Grid X")
                ax1.set_ylabel("Grid Y")
                ax1.set_title("2D Spatial Habitat & Organism Distribution")
                ax1.set_aspect("equal")
                ax1.grid(True, alpha=0.2)
                solara.FigureMatplotlib(fig1)
                plt.close(fig1)

        # ---- Panel 2: State Distribution (Current) ----
        with solara.Card("2. State Distribution (Current)"):
            if m is not None:
                state_values = [agent.organism.state for agent in m.agents]
                if state_values:
                    states, state_counts = np.unique(state_values, return_counts=True)
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.bar(states.astype(str), state_counts, color="steelblue")
                    ax2.set_xlabel("State")
                    ax2.set_ylabel("Current organisms")
                    ax2.set_title("State Distribution (Current)")
                    ax2.tick_params(axis="x", rotation=45)
                    solara.FigureMatplotlib(fig2)
                    plt.close(fig2)

        # ---- Panel 3: Environmental Driver Distributions ----
        with solara.Card("3. Environmental Driver Distributions"):
            if m is not None:
                _, _, depth_values, temp_values, hsi_values = _viewport_environment(
                    np.linspace(-66.5, -57.5, 26), np.linspace(42.0, 48.5, 20)
                )
                fig3, axes3 = plt.subplots(1, 3, figsize=(10, 3.5))
                axes3[0].hist(depth_values.ravel(), bins=20, color="steelblue", alpha=0.8)
                axes3[0].set_xlabel("Depth (m)")
                axes3[1].hist(temp_values.ravel(), bins=20, color="darkorange", alpha=0.8)
                axes3[1].set_xlabel("Bottom temperature (°C)")
                axes3[2].hist(hsi_values.ravel(), bins=20, color="seagreen", alpha=0.8)
                axes3[2].set_xlabel("HSI")
                for axis in axes3:
                    axis.set_ylabel("Cells in viewport")
                local_status = environmental_input_status()
                data_loaded = local_status["nwa_bathymetry"] or local_status["woa23_temperature"]
                fig3.suptitle(
                    "Environmental distributions across map viewport"
                    if data_loaded
                    else "Illustrative environmental distributions (local grids unavailable)"
                )
                fig3.tight_layout()
                solara.FigureMatplotlib(fig3)
                plt.close(fig3)

        # ---- Panel 4: Abundance Trajectories ----
        with solara.Card("4. Abundance Trajectories"):
            if len(history) > 1:
                fig4, ax4 = plt.subplots(figsize=(8, 4))
                steps = [h["step"] for h in history]
                pops = [max(1, h["n_agents"]) for h in history]
                ax4.plot(steps, np.log10(pops), color="royalblue", lw=2.0, label="Total Organisms")
                ax4.set_xlabel("Time Step")
                ax4.set_ylabel("log10(Count)")
                ax4.set_title("Abundance Trajectories")
                ax4.grid(True, alpha=0.3)
                ax4.legend(frameon=False)
                solara.FigureMatplotlib(fig4)
                plt.close(fig4)

        # ---- Panel 5: Biomass, Catch & Macro-Models ----
        with solara.Card("5. Biomass, Catch & Macro-Models"):
            if len(history) > 1:
                fig5, ax5 = plt.subplots(figsize=(8, 4))
                steps = [h["step"] for h in history]
                biomass = [h["biomass_kg"] for h in history]
                catches = [h["catch_total"] for h in history]
                ax5.plot(steps, biomass, "b-", lw=2, label="Biomass (kg)")
                ax5.plot(steps, catches, "r--", lw=2, label="Cumulative catch")
                ax5.set_xlabel("Time Step")
                ax5.set_ylabel("Biomass / organisms")
                ax5.set_title("Biomass, Catch & Macro-Models")
                ax5.grid(True, alpha=0.3)
                ax5.legend()
                solara.FigureMatplotlib(fig5)
                plt.close(fig5)

        # ---- Panel 6: Living Stage Structure ----
        with solara.Card("6. Living Stage Structure"):
            if m is not None:
                stage_values = [agent.organism.stage.value for agent in m.agents]
                if stage_values:
                    stages, stage_counts = np.unique(stage_values, return_counts=True)
                    fig6, ax6 = plt.subplots(figsize=(8, 4))
                    ax6.bar(stages.astype(str), stage_counts, color="steelblue")
                    ax6.set_xlabel("Developmental Stage")
                    ax6.set_ylabel("Living organisms")
                    ax6.set_title("Living Stage Structure (current)")
                    ax6.tick_params(axis="x", rotation=45)
                    solara.FigureMatplotlib(fig6)
                    plt.close(fig6)

        # ---- Panel 7: Stock-Recruitment Relationship ----
        with solara.Card("7. Stock-Recruitment Relationship"):
            if len(history) > 2:
                fig7, ax7 = plt.subplots(figsize=(6, 6))
                stocks = [h["mature_females"] for h in history]
                recruits = [h["juvenile_recruits"] for h in history]
                ax7.scatter(stocks, recruits, c="blue", s=10, alpha=0.5)
                ax7.set_xlabel("Mature females")
                ax7.set_ylabel("Juvenile recruits")
                ax7.set_title("Stock-Recruitment Relationship")
                ax7.grid(True, alpha=0.3)
                solara.FigureMatplotlib(fig7)
                plt.close(fig7)

        # ---- Panel 8: Size Frequency Spectra by Sex & Maturity ----
        with solara.Card("8. Size Frequency Spectra by Sex & Maturity"):
            if m is not None:
                organisms = [agent.organism for agent in m.agents]
                if organisms:
                    fig8, ax8 = plt.subplots(figsize=(8, 5))
                    for sex, color in [("male", "blue"), ("female", "red")]:
                        for mature, style in [(False, "-"), (True, "--")]:
                            subset = [
                                organism.carapace_width
                                for organism in organisms
                                if organism.sex.value == sex and organism.is_mature == mature
                            ]
                            if subset:
                                ax8.hist(
                                    subset, bins=20, alpha=0.35, color=color, histtype="step",
                                    linestyle=style, label=f"{sex}, {'mature' if mature else 'immature'}",
                                    density=True,
                                )
                    ax8.set_xlabel("Carapace Width (mm)")
                    ax8.set_ylabel("Density")
                    ax8.set_title("Current Size Frequency by Sex & Maturity")
                    ax8.legend()
                    ax8.grid(True, alpha=0.3)
                    solara.FigureMatplotlib(fig8)
                    plt.close(fig8)

        # ---- Panel 9: Maturity Ogives & Selectivity ----
        with solara.Card("9. Maturity Ogives & Selectivity"):
            if m is not None:
                sp = m.params.species
                fishery = m.params.fishery
                sizes = np.linspace(20.0, max(sp.max_size_male, sp.max_size_female), 300)
                def ogive(l50, slope):
                    values = 1.0 / (1.0 + np.exp(-slope * np.log(np.maximum(sizes, 1.0) / l50)))
                    values[sizes < sp.maturity_size_gate] = 0.0
                    return values
                fig9, ax9 = plt.subplots(figsize=(8, 5))
                ax9.plot(
                    sizes, ogive(sp.cw50_maturity, sp.maturity_slope_male),
                    color="blue", lw=2, label="Male maturity",
                )
                ax9.plot(
                    sizes, ogive(sp.cw50_maturity_female, sp.maturity_slope_female),
                    color="red", lw=2, label="Female maturity",
                )
                selectivity = 1.0 / (
                    1.0 + np.exp(
                        -(sizes - sp.selectivity_s50) / max(sp.selectivity_s50_sigma, 1e-6)
                    )
                )
                ax9.plot(sizes, selectivity, "g--", lw=2, label="Selectivity")
                ax9.axvline(fishery.legal_size_mm, color="black", linestyle=":", label="Legal size")
                ax9.set_xlabel("Carapace Width (mm)")
                ax9.set_ylabel("Probability")
                ax9.set_title("Maturity Ogives & Selectivity")
                ax9.set_ylim(0.0, 1.0)
                ax9.legend()
                ax9.grid(True, alpha=0.3)
                solara.FigureMatplotlib(fig9)
                plt.close(fig9)

        # ---- Panel 10: Multi-Attribute Calibration Criteria ----
        with solara.Card("10. Multi-Attribute Calibration Criteria"):
            if len(history) > 1:
                fig10, ax10 = plt.subplots(figsize=(8, 4))
                steps = list(range(1, len(history) + 1))
                pops = [h["n_agents"] for h in history]
                ax10.plot(steps, pops, "b-", lw=2, label="Population")
                ax10.axhline(y=100, color="r", linestyle="--", label="Target")
                ax10.set_xlabel("Time Step")
                ax10.set_ylabel("Population")
                ax10.set_title("Multi-Attribute Calibration Criteria")
                ax10.legend()
                ax10.grid(True, alpha=0.3)
                solara.FigureMatplotlib(fig10)
                plt.close(fig10)

        # ---- Raw Data Table (matching HTML reference) ----
        with solara.Card("Simulation Output Data"):
            if len(history) > 0:
                solara.DataFrame(
                    data={
                        "Time": [h.get("step", i + 1) for i, h in enumerate(history)],
                        "Foraging": [h.get("n_foraging", h["n_agents"]) for h in history],
                        "Resting": [h.get("n_resting", 0) for h in history],
                        "Molting": [h.get("n_molting", 0) for h in history],
                        "Dead": [h.get("n_dead", 0) for h in history],
                    }
                )

    return


Page = RealtimeDashboard
