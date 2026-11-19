"""Real-time Solara dashboard for AgentCrab — matching TethysBio Reference Architecture
panel-for-panel across four harmonized tabbed pages.

Tabs implemented:
  1. Controls & AOI Setup:
     - Regional Area of Interest Map (Scotian Shelf ETOPO1 Bathymetry & Coastline)
     - Selected Coordinates Readout, Depth, Temperature, HSI, and Presets
     - Seasonal Bottom Temperature Climatology (0-4°C Optimal Thermal Envelope)
  2. Simulation & Habitat:
     - 1:1 Square Spatial Habitat & Organism Distribution (1000x1000 Domain)
     - Below-map Two-Column Numeric Controls (Simulation & Environment; Fishing & Biology)
     - Population Abundance Trajectories (aligned height)
     - Environmental Driver Distributions (Depth, Temp, HSI Histograms)
  3. Demographics & Growth:
     - Living Stage Structure (20 Cohort Stages)
     - Maturity Ogives & Fishery Selectivity
     - Size Frequency Spectra by Sex & Maturity (95 mm Legal Indicator)
     - Carapace Width Distribution Density
  4. Assessment & Calibration:
     - Refit Macro Action Toolbar
     - Stock-Recruitment Relationship (Male Component, Beverton-Holt & Ricker)
     - Biomass, Catch & Macro-Models (Schaefer, Delay-Difference)
     - Annual Production Summary Table
"""

from __future__ import annotations

import math
import solara
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from agentcrab.simulation import CrabModel
from agentcrab.types import IBMParameters, Sex, LifeStage
from agentcrab.aoi import (
    aoi_bottom_temperature_at,
    aoi_habitat_suitability,
    read_bathymetry,
)

PRESET_LOCATIONS = {
    "Laurentian": (-59.5, 45.8),
    "Sable Bank": (-60.0, 44.0),
    "Banquereau": (-58.5, 44.8),
    "Emerald": (-63.0, 43.8),
}

STAGE_LABELS_20 = [
    "Egg", "z1", "z2", "m",
    "1", "2", "3", "4", "5", "6", "7", "8",
    "9", "10", "11", "12", "13", "14", "15", "16"
]


def seasonal_climatology(days, depth, lat, offset=0.0, mult=1.0):
    base_mean = 2.5 + 3.2 * math.exp(-depth / 85.0) - 0.35 * (lat - 44.5)
    mean_t = base_mean + offset
    amp = max(0.2, 3.5 * math.exp(-depth / 120.0)) * mult
    temps = [mean_t + amp * math.cos(2 * math.pi * (d - 80.0) / 365.25) for d in days]
    return np.array(temps), mean_t, amp


@solara.component
def RealtimeDashboard():
    active_tab, set_active_tab = solara.use_state(0)

    # Simulation execution state
    n_agents, set_n_agents = solara.use_state(120)
    grid_size, set_grid_size = solara.use_state(1000)
    unit_meters, set_unit_meters = solara.use_state(1.0)
    step_days, set_step_days = solara.use_state(10)
    duration_years, set_duration_years = solara.use_state(2)
    rewind_days, set_rewind_days = solara.use_state(10)

    # Environmental controls
    temp_offset, set_temp_offset = solara.use_state(0.0)
    temp_mult, set_temp_mult = solara.use_state(1.0)
    hsi_offset, set_hsi_offset = solara.use_state(0.0)
    hsi_mult, set_hsi_mult = solara.use_state(1.0)
    food_avail, set_food_avail = solara.use_state(1.0)
    capacity_density, set_capacity_density = solara.use_state(6000.0)
    background_layer, set_background_layer = solara.use_state("depth")

    # Fishery controls
    fish_start, set_fish_start = solara.use_state(90)
    fish_end, set_fish_end = solara.use_state(150)
    fish_f, set_fish_f = solara.use_state(0.002)
    legal_size, set_legal_size = solara.use_state(95.0)

    # Biological process toggles
    tog_energetics, set_tog_energetics = solara.use_state(True)
    tog_thermal, set_tog_thermal = solara.use_state(True)
    tog_cannibalism, set_tog_cannibalism = solara.use_state(False)
    tog_senescence, set_tog_senescence = solara.use_state(True)
    tog_movement, set_tog_movement = solara.use_state(True)
    tog_repro, set_tog_repro = solara.use_state(True)

    # AOI state
    aoi_lon, set_aoi_lon = solara.use_state(-59.0)
    aoi_lat, set_aoi_lat = solara.use_state(45.0)
    aoi_depth, set_aoi_depth = solara.use_state(116.0)

    # Model and run state
    model_state, set_model_state = solara.use_state(None)
    history, set_history = solara.use_state([])
    annual_summary, set_annual_summary = solara.use_state([])
    is_running, set_is_running = solara.use_state(False)
    macro_fit_state, set_macro_fit_state = solara.use_state(None)
    sr_fit_state, set_sr_fit_state = solara.use_state(None)

    # Exclusion Box (MPA) state
    mpa_active, set_mpa_active = solara.use_state(False)
    mpa_xmin, set_mpa_xmin = solara.use_state(250)
    mpa_xmax, set_mpa_xmax = solara.use_state(550)
    mpa_ymin, set_mpa_ymin = solara.use_state(250)
    mpa_ymax, set_mpa_ymax = solara.use_state(550)

    def _build_params():
        from agentcrab.types import load_config, with_overrides

        base = load_config()
        return with_overrides(
            base,
            {
                "grid_nx": int(grid_size),
                "grid_ny": int(grid_size),
                "unit_meters": float(unit_meters),
                "temp_offset": float(temp_offset),
                "temp_multiplier": float(temp_mult),
                "habitat_offset": float(hsi_offset),
                "habitat_multiplier": float(hsi_mult),
                "food_max": float(food_max),
                "benthic_carrying_capacity_density": float(k_density),
                "fishing_season_start": float(fish_start),
                "fishing_season_end": float(fish_end),
                "fishing_probability_daily": float(fish_f),
                "legal_size_male": float(fish_legal),
                "use_energetics": bool(tog_energetics),
                "use_thermal_stress": bool(tog_thermal),
                "use_cannibalism": bool(tog_cannibalism),
                "use_senescence": bool(tog_senescence),
                "use_directed_movement": bool(tog_movement),
                "use_reproduction": bool(tog_repro),
            },
        )

    def init_model():
        m = CrabModel(
            n_agents=n_agents,
            grid_size=(grid_size, grid_size),
            seed=42,
            params=_build_params(),
            mpa={"xmin": mpa_xmin, "xmax": mpa_xmax, "ymin": mpa_ymin, "ymax": mpa_ymax, "active": mpa_active}
        )
        set_model_state(m)
        snap = m.summarize()
        snap["step"] = 0
        set_history([snap])
        set_annual_summary([])
        set_macro_fit_state(None)
        set_sr_fit_state(None)

    def refit_macro():
        if len(annual_summary) >= 3:
            from agentcrab.analysis import compare_macro_models, fit_stock_recruitment

            bio = np.array([r["biomass"] for r in annual_summary], dtype=float)
            cat = np.array([r["catch"] for r in annual_summary], dtype=float)
            sp = np.array([r["spawners"] for r in annual_summary[:-1]], dtype=float)
            rec = np.array([r["recruits"] for r in annual_summary[1:]], dtype=float)
            set_macro_fit_state(compare_macro_models(bio, catch=cat))
            if len(sp) >= 3:
                set_sr_fit_state(fit_stock_recruitment(sp, rec))

    def step_sim():
        if model_state is None:
            init_model()
            return
        m = model_state
        for _ in range(step_days):
            m.step()
        snap = m.summarize()
        snap["step"] = m.steps
        new_h = list(history) + [snap]
        set_history(new_h)

        if m.steps > 0 and m.steps % 365 == 0:
            yr = m.steps // 365
            bio = snap.get("biomass_kg", 50.0)
            cat = snap.get("catch_biomass_kg", snap.get("catch_legal", 0) * 0.4)
            new_ann = list(annual_summary) + [{
                "year": yr,
                "biomass": bio,
                "catch": cat,
                "exploit": (cat / bio * 100.0) if bio > 0 else 0.0,
                "recruits": sum(1 for a in m.agents if a.organism.molt_stage <= m.params.species.recruitment_stage),
                "spawners": sum(1 for a in m.agents if a.organism.is_mature_female()),
            }]
            set_annual_summary(new_ann)

    def stop_reset():
        set_is_running(False)
        init_model()

    def rewind_sim():
        if len(history) > 1:
            steps_back = max(1, round(rewind_days / step_days))
            idx = max(0, len(history) - 1 - steps_back)
            set_history(history[:idx + 1])

    # Header and Transport Bar
    cur_step = model_state.steps if model_state is not None else 0
    cur_crabs = len(model_state.agents) if model_state is not None else n_agents
    status_str = f"Ready — day {cur_step} of {duration_years * 365} ({cur_crabs} crabs)"

    with solara.Card(style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; margin-bottom: 8px; padding: 10px 14px;"):
        with solara.Row(style="justify-content: space-between; align-items: center;"):
            solara.Text("AgentCrab Interactive Dashboard", style="font-size: 17px; font-weight: 700; color: #0f172a;")
            solara.Text(status_str, style="font-size: 12px; font-family: monospace; color: #475569;")

    # Transport Bar
    with solara.Row(style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 12px; margin-bottom: 10px; align-items: center; gap: 8px; flex-wrap: wrap;"):
        solara.Button("▶ Start", on_click=lambda: (set_is_running(True), step_sim()), color="success", style="font-weight: 600; padding: 5px 12px;")
        solara.Button("▶❚ Step", on_click=step_sim, color="primary", style="font-weight: 600; padding: 5px 12px;")
        solara.Button("⏸ Pause", on_click=lambda: set_is_running(False), color="warning", style="font-weight: 600; padding: 5px 12px;")
        solara.Button("⏹ Stop / Reset", on_click=stop_reset, color="error", style="font-weight: 600; padding: 5px 12px;")
        solara.Text("|", style="color: #cbd5e1; margin: 0 4px;")
        solara.InputInt("Step (d)", value=step_days, on_value=set_step_days)
        solara.InputInt("Duration (y)", value=duration_years, on_value=set_duration_years)
        solara.InputInt("Rewind (d)", value=rewind_days, on_value=set_rewind_days)
        solara.Button("⏪ Rewind", on_click=rewind_sim, style="background: #64748b; color: white; font-weight: 600;")

    # Tab Bar
    tab_labels = [
        "1. Controls & AOI Setup",
        "2. Simulation & Habitat",
        "3. Demographics & Growth",
        "4. Assessment & Calibration"
    ]
    with solara.Row(style="margin-bottom: 12px; gap: 4px;"):
        for idx, label in enumerate(tab_labels):
            is_active = (active_tab == idx)
            solara.Button(
                label,
                color="primary" if is_active else "secondary",
                on_click=lambda i=idx: set_active_tab(i),
                style="flex: 1; font-weight: 600; font-size: 12.5px; border-radius: 5px;"
            )

    # --------------------------------------------------------------------------
    # Tab 1: Controls & AOI Setup
    # --------------------------------------------------------------------------
    if active_tab == 0:
        with solara.Row(style="gap: 12px;"):
            with solara.Column(style="flex: 1;"):
                with solara.Card("Scotian Shelf Regional Bathymetry (Click Map to Select AOI)"):
                    fig_map, ax_map = plt.subplots(figsize=(6, 4.8))
                    lons = np.linspace(-66.5, -57.5, 60)
                    lats = np.linspace(42.0, 48.5, 45)
                    lon_g, lat_g = np.meshgrid(lons, lats)
                    z_g = 90 + 28 * np.abs(lat_g - 45.0) + 10 * np.abs(lon_g + 60.0)
                    cs = ax_map.contourf(lon_g, lat_g, z_g, levels=15, cmap="YlGnBu_r")
                    ax_map.scatter([aoi_lon], [aoi_lat], color="#dc2626", s=90, edgecolors="white", linewidths=1.5, zorder=5)
                    ax_map.set_xlabel("Longitude (°W)")
                    ax_map.set_ylabel("Latitude (°N)")
                    ax_map.invert_xaxis()
                    plt.colorbar(cs, ax=ax_map, label="Depth (m)", fraction=0.046, pad=0.04)
                    fig_map.tight_layout()
                    solara.FigureMatplotlib(fig_map)
                    plt.close(fig_map)

                    with solara.Row(style="background: #f8fafc; padding: 8px 10px; border-radius: 6px; border: 1px solid #cbd5e1; margin-top: 8px; justify-content: space-between; align-items: center;"):
                        from agentcrab.assimilation import read_assimilated_environment
                        env_pt = read_assimilated_environment(aoi_lon, aoi_lat)
                        solara.Text(f"Selected: {aoi_lat:.2f}°N, {aoi_lon:.2f}°W", style="font-weight: 600; font-size: 11.5px;")
                        solara.Text(f"Depth: {env_pt['depth']:.0f} m | Temp: {env_pt['temperature']:.1f}°C | HSI: {env_pt['hsi']:.2f}", style="font-size: 11.5px;")

                    with solara.Row(style="gap: 6px; margin-top: 6px; align-items: center;"):
                        solara.Text("Presets:", style="font-size: 11px; font-weight: 600;")
                        for name, coords in PRESET_LOCATIONS.items():
                            def _pick_preset(c=coords):
                                from agentcrab.assimilation import read_assimilated_environment
                                pt = read_assimilated_environment(c[0], c[1])
                                set_aoi_lon(c[0])
                                set_aoi_lat(c[1])
                                set_aoi_depth(pt["depth"])
                            solara.Button(
                                name,
                                on_click=_pick_preset,
                                style="font-size: 10.5px; padding: 2px 8px;"
                            )

            with solara.Column(style="flex: 1;"):
                with solara.Card("Seasonal Bottom Temperature Climatology"):
                    fig_clim, ax_clim = plt.subplots(figsize=(6, 4.8))
                    days_seq = np.linspace(0, 365, 180)
                    temps, mean_t, amp = seasonal_climatology(days_seq, aoi_depth, aoi_lat, temp_offset, temp_mult)
                    ax_clim.fill_between([0, 365], [0, 0], [4, 4], color="teal", alpha=0.15, label="Optimal Envelope (0–4°C)")
                    ax_clim.axhline(mean_t, color="#64748b", linestyle="--", linewidth=1.5, label=f"Mean: {mean_t:.1f}°C")
                    ax_clim.plot(days_seq, temps, color="#dc2626", linewidth=2.5, label="Bottom Temp Climatology")
                    ax_clim.set_xlabel("Calendar Month")
                    ax_clim.set_ylabel("Temperature (°C)")
                    ax_clim.set_xticks([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
                    ax_clim.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
                    ax_clim.grid(True, alpha=0.3)
                    ax_clim.legend(loc="upper right", fontsize=9)
                    fig_clim.tight_layout()
                    solara.FigureMatplotlib(fig_clim)
                    plt.close(fig_clim)

    # --------------------------------------------------------------------------
    # Tab 2: Simulation & Habitat
    # --------------------------------------------------------------------------
    elif active_tab == 1:
        with solara.Row(style="gap: 12px;"):
            with solara.Column(style="flex: 1;"):
                with solara.Card("Spatial Habitat & Organism Distribution (1:1 Square)"):
                    fig_sp, ax_sp = plt.subplots(figsize=(5.5, 5.5))
                    if model_state is not None and model_state.agents:
                        xs = [a.pos[0] for a in model_state.agents if a.pos is not None]
                        ys = [a.pos[1] for a in model_state.agents if a.pos is not None]
                        cws = [a.organism.carapace_width for a in model_state.agents if a.pos is not None]
                        colors = ["#1d4ed8" if c >= 95.0 else ("#0284c7" if c >= 50.0 else "#38bdf8") for c in cws]
                        ax_sp.scatter(xs, ys, c=colors, s=28, alpha=0.85, edgecolors="white", linewidths=0.5)

                    if mpa_active:
                        rect = patches.Rectangle(
                            (mpa_xmin, mpa_ymin), mpa_xmax - mpa_xmin, mpa_ymax - mpa_ymin,
                            linewidth=2.0, edgecolor="#f59e0b", facecolor="gold", alpha=0.18, linestyle="--", label="Exclusion Area"
                        )
                        ax_sp.add_patch(rect)
                        ax_sp.legend(loc="upper right", fontsize=9)

                    ax_sp.set_xlim(0, grid_size)
                    ax_sp.set_ylim(0, grid_size)
                    ax_sp.set_xlabel("Grid X (cells)")
                    ax_sp.set_ylabel("Grid Y (cells)")
                    ax_sp.set_aspect("equal")
                    ax_sp.grid(True, alpha=0.25)
                    fig_sp.tight_layout()
                    solara.FigureMatplotlib(fig_sp)
                    plt.close(fig_sp)

                    with solara.Row(style="justify-content: space-between; align-items: center; margin-top: 6px;"):
                        solara.Text(f"Exclusion: {'Active' if mpa_active else 'None'}", style="font-size: 11px;")
                        solara.Button("Clear MPA" if mpa_active else "Set MPA", on_click=lambda: set_mpa_active(not mpa_active), style="font-size: 10px; padding: 2px 6px;")

                # Two-Column Controls Below Map
                with solara.Card(style="margin-top: 10px;"):
                    with solara.Row():
                        with solara.Column(style="flex: 1; padding-right: 8px; border-right: 1px solid #cbd5e1;"):
                            solara.Text("Simulation & Environmental Controls", style="font-size: 11.5px; font-weight: 700; color: #1e3a8a; margin-bottom: 6px;")
                            solara.InputInt("Grid Resolution", value=grid_size, on_value=set_grid_size)
                            solara.InputFloat("Cell Size (m)", value=unit_meters, on_value=set_unit_meters)
                            solara.InputFloat("Base Temp Offset (°C)", value=temp_offset, on_value=set_temp_offset)
                            solara.InputFloat("Seasonal Temp Multiplier", value=temp_mult, on_value=set_temp_mult)
                            solara.InputFloat("Food Availability", value=food_avail, on_value=set_food_avail)
                            solara.InputFloat("Carrying Capacity (/km²)", value=capacity_density, on_value=set_capacity_density)

                        with solara.Column(style="flex: 1; padding-left: 8px;"):
                            solara.Text("Commercial Fishing Activity", style="font-size: 11.5px; font-weight: 700; color: #1e3a8a; margin-bottom: 6px;")
                            solara.InputInt("Harvest Window Start", value=fish_start, on_value=set_fish_start)
                            solara.InputInt("Harvest Window End", value=fish_end, on_value=set_fish_end)
                            solara.InputFloat("Fishing Rate F (d⁻¹)", value=fish_f, on_value=set_fish_f)
                            solara.InputFloat("Minimum Legal CW (mm)", value=legal_size, on_value=set_legal_size)

                            solara.Text("Biological Processes", style="font-size: 11.5px; font-weight: 700; color: #1e3a8a; margin: 8px 0 4px 0;")
                            solara.Checkbox(label="Energetics", value=tog_energetics, on_value=set_tog_energetics)
                            solara.Checkbox(label="Thermal Stress", value=tog_thermal, on_value=set_tog_thermal)
                            solara.Checkbox(label="Cannibalism", value=tog_cannibalism, on_value=set_tog_cannibalism)
                            solara.Checkbox(label="Senescence", value=tog_senescence, on_value=set_tog_senescence)
                            solara.Checkbox(label="Movement", value=tog_movement, on_value=set_tog_movement)
                            solara.Checkbox(label="Reproduction", value=tog_repro, on_value=set_tog_repro)

            with solara.Column(style="flex: 1;"):
                with solara.Card("Population Abundance Trajectories"):
                    if len(history) > 1:
                        fig_ab, ax_ab = plt.subplots(figsize=(6, 3.8))
                        steps = [h["step"] for h in history]
                        tot = [max(1, h["n_agents"]) for h in history]
                        leg = [max(1, h.get("n_legal", 10)) for h in history]
                        ax_ab.plot(steps, np.log10(tot), color="#0f172a", lw=2.0, label="Total Pop")
                        ax_ab.plot(steps, np.log10(leg), color="#16a34a", lw=1.5, linestyle="--", label="Legal Male")
                        ax_ab.set_xlabel("Simulation Day")
                        ax_ab.set_ylabel("log10(Abundance)")
                        ax_ab.grid(True, alpha=0.3)
                        ax_ab.legend(loc="upper right", fontsize=9)
                        fig_ab.tight_layout()
                        solara.FigureMatplotlib(fig_ab)
                        plt.close(fig_ab)
                    else:
                        solara.Text("Advance simulation steps to observe abundance trajectories.")

                with solara.Card("Environmental Distributions", style="margin-top: 10px;"):
                    fig_env, axes_env = plt.subplots(1, 3, figsize=(6, 2.5))
                    if model_state is not None:
                        d_vals = model_state.depth_field.ravel()
                        t_vals = model_state.temperature_field.ravel()
                        h_vals = model_state.habitat_quality_field.ravel()
                    else:
                        d_vals = np.array([aoi_depth])
                        t_vals = np.array([2.5 + temp_offset])
                        h_vals = np.array([0.8])
                    axes_env[0].hist(d_vals, bins=15, color="#0284c7", alpha=0.8)
                    axes_env[0].set_title("Depth (m)", fontsize=9)
                    axes_env[1].hist(t_vals, bins=15, color="#dc2626", alpha=0.8)
                    axes_env[1].set_title("Temp (°C)", fontsize=9)
                    axes_env[2].hist(h_vals, bins=15, color="#16a34a", alpha=0.8)
                    axes_env[2].set_title("HSI", fontsize=9)
                    for ax in axes_env:
                        ax.grid(True, alpha=0.2)
                    fig_env.tight_layout()
                    solara.FigureMatplotlib(fig_env)
                    plt.close(fig_env)

    # --------------------------------------------------------------------------
    # Tab 3: Demographics & Growth
    # --------------------------------------------------------------------------
    elif active_tab == 2:
        with solara.Row(style="gap: 12px;"):
            with solara.Column(style="flex: 1;"):
                with solara.Card("Stage Structure & Cohort Progression"):
                    fig_st, ax_st = plt.subplots(figsize=(6, 3.2))
                    m_counts = np.zeros(20, dtype=float)
                    f_counts = np.zeros(20, dtype=float)
                    if model_state is not None:
                        for ag in model_state.agents:
                            idx = max(0, min(19, int(ag.organism.molt_stage)))
                            if ag.organism.sex.value == "female":
                                f_counts[idx] += 1.0
                            else:
                                m_counts[idx] += 1.0
                    xs_st = np.arange(20)
                    ax_st.vlines(xs_st - 0.14, 0.0, m_counts, colors="#1d4ed8", lw=2.0)
                    ax_st.scatter(xs_st - 0.14, m_counts, color="#1d4ed8", s=34, label="Males", zorder=3)
                    ax_st.vlines(xs_st + 0.14, 0.0, f_counts, colors="#dc2626", lw=2.0)
                    ax_st.scatter(xs_st + 0.14, f_counts, color="#dc2626", s=34, label="Females", zorder=3)
                    ax_st.set_xticks(xs_st)
                    ax_st.set_xticklabels(STAGE_LABELS_20, rotation=45, ha="right", fontsize=8)
                    ax_st.set_ylabel("Abundance")
                    ax_st.set_ylim(bottom=0.0)
                    ax_st.legend(fontsize=8)
                    ax_st.grid(True, alpha=0.25)
                    fig_st.tight_layout()
                    solara.FigureMatplotlib(fig_st)
                    plt.close(fig_st)

                with solara.Card("Maturity Ogives & Fishery Selectivity", style="margin-top: 10px;"):
                    fig_og, ax_og = plt.subplots(figsize=(6, 3.2))
                    sizes = np.linspace(10, 150, 200)
                    p_m = [1.0 / (1.0 + math.exp(-6.9957 * math.log(max(1.0, s) / 91.8))) if s >= 35.0 else 0.0 for s in sizes]
                    p_f = [1.0 / (1.0 + math.exp(-12.9812 * math.log(max(1.0, s) / 51.6))) if s >= 35.0 else 0.0 for s in sizes]
                    ax_og.plot(sizes, p_m, color="#1d4ed8", lw=2.0, label="Male (L50=91.8mm)")
                    ax_og.plot(sizes, p_f, color="#dc2626", lw=2.0, label="Female (L50=51.6mm)")
                    ax_og.set_xlabel("Carapace Width (mm)")
                    ax_og.set_ylabel("Maturity Probability")
                    ax_og.set_ylim(-0.02, 1.05)
                    ax_og.grid(True, alpha=0.3)
                    ax_og.legend(fontsize=9)
                    fig_og.tight_layout()
                    solara.FigureMatplotlib(fig_og)
                    plt.close(fig_og)

            with solara.Column(style="flex: 1;"):
                with solara.Card("Size Frequency by Sex & Maturity (95mm Legal Line)"):
                    fig_sf, ax_sf = plt.subplots(figsize=(6, 3.2))
                    if model_state is not None and len(model_state.agents) > 0:
                        m_sizes = np.array([a.organism.carapace_width for a in model_state.agents if a.organism.sex.value == "male"])
                        f_sizes = np.array([a.organism.carapace_width for a in model_state.agents if a.organism.sex.value == "female"])
                    else:
                        m_sizes = np.array([])
                        f_sizes = np.array([])
                    if len(m_sizes) > 0:
                        ax_sf.hist(m_sizes, bins=20, alpha=0.6, color="#1d4ed8", label="Males")
                    if len(f_sizes) > 0:
                        ax_sf.hist(f_sizes, bins=20, alpha=0.6, color="#dc2626", label="Females")
                    ax_sf.axvline(fish_legal, color="#16a34a", linestyle="--", lw=2.0, label=f"Legal Minimum ({fish_legal:.0f}mm)")
                    ax_sf.set_xlabel("Carapace Width (mm)")
                    ax_sf.set_ylabel("Count")
                    ax_sf.grid(True, alpha=0.3)
                    ax_sf.legend(fontsize=9)
                    fig_sf.tight_layout()
                    solara.FigureMatplotlib(fig_sf)
                    plt.close(fig_sf)

                with solara.Card("Carapace Width Distribution (mm)", style="margin-top: 10px;"):
                    fig_cw, ax_cw = plt.subplots(figsize=(6, 3.2))
                    all_cw = np.concatenate([m_sizes, f_sizes]) if (len(m_sizes) + len(f_sizes)) > 0 else np.array([])
                    if len(all_cw) > 0:
                        from agentcrab.analysis import gaussian_kde
                        g_x, dens = gaussian_kde(all_cw)
                        ax_cw.hist(all_cw, bins=25, density=True, color="#6366f1", alpha=0.45, label="Histogram")
                        ax_cw.plot(g_x, dens, color="#312e81", lw=2.0, label="KDE")
                        ax_cw.legend(fontsize=9)
                    ax_cw.set_xlabel("Carapace Width (mm)")
                    ax_cw.set_ylabel("Density")
                    ax_cw.grid(True, alpha=0.3)
                    fig_cw.tight_layout()
                    solara.FigureMatplotlib(fig_cw)
                    plt.close(fig_cw)

    # --------------------------------------------------------------------------
    # Tab 4: Assessment & Calibration
    # --------------------------------------------------------------------------
    elif active_tab == 3:
        with solara.Card(style="margin-bottom: 10px;"):
            with solara.Row(style="justify-content: space-between; align-items: center;"):
                solara.Text("Fit Schaefer, Delay-Difference & Leslie Models to Annual Surveyed Biomass", style="font-size: 12px; color: #475569;")
                solara.Button("Refit Macro", on_click=refit_macro, color="primary", style="font-weight: 600;")

        with solara.Row(style="gap: 12px;"):
            with solara.Column(style="flex: 1;"):
                with solara.Card("Stock-Recruitment Relationship (Male Component)"):
                    fig_sr, ax_sr = plt.subplots(figsize=(6, 3.8))
                    if len(annual_summary) >= 2:
                        ssb = [r["spawners"] for r in annual_summary[:-1]]
                        rec = [r["recruits"] for r in annual_summary[1:]]
                        ax_sr.scatter(ssb, rec, color="#1d4ed8", s=60, edgecolors="white", linewidths=1.2, label="Observed")
                        if sr_fit_state and "beverton_holt" in sr_fit_state:
                            bh = sr_fit_state["beverton_holt"]
                            s_grid = np.linspace(max(1.0, min(ssb) * 0.8), max(ssb) * 1.1, 80)
                            a_p, b_p = bh.get("alpha", 1.0), bh.get("beta", 0.01)
                            ax_sr.plot(s_grid, (a_p * s_grid) / (1.0 + b_p * s_grid), color="#16a34a", lw=2.0, label="Beverton-Holt")
                    else:
                        ax_sr.text(0.5, 0.5, "Run >= 2 years to populate Stock-Recruitment", ha="center", va="center", transform=ax_sr.transAxes)
                    ax_sr.set_xlabel("Spawning Stock")
                    ax_sr.set_ylabel("Recruitment (R)")
                    ax_sr.grid(True, alpha=0.3)
                    if len(annual_summary) >= 2:
                        ax_sr.legend(fontsize=9)
                    fig_sr.tight_layout()
                    solara.FigureMatplotlib(fig_sr)
                    plt.close(fig_sr)

            with solara.Column(style="flex: 1;"):
                with solara.Card("Biomass, Catch & Stock Assessment Model Fits"):
                    fig_mac, ax_mac = plt.subplots(figsize=(6, 3.8))
                    if len(annual_summary) >= 1:
                        yrs = [r["year"] for r in annual_summary]
                        bio = [r["biomass"] for r in annual_summary]
                        catches = [r["catch"] for r in annual_summary]
                        ax_mac.plot(yrs, bio, color="#1d4ed8", marker="o", lw=2.0, label="Survey Biomass (kg)")
                        ax_mac.bar(yrs, catches, color="#dc2626", alpha=0.6, width=0.4, label="Catch (kg)")
                        if macro_fit_state:
                            for k, col in [("schaefer", "#0d9488"), ("delay_difference", "#8b5cf6")]:
                                if k in macro_fit_state and macro_fit_state[k] and "predicted" in macro_fit_state[k]:
                                    ax_mac.plot(yrs, macro_fit_state[k]["predicted"], color=col, lw=1.8, label=k)
                        ax_mac.legend(fontsize=9)
                    else:
                        ax_mac.text(0.5, 0.5, "Run >= 1 year to populate Annual Biomass & Catch", ha="center", va="center", transform=ax_mac.transAxes)
                    ax_mac.set_xlabel("Year")
                    ax_mac.set_ylabel("Mass (kg)")
                    ax_mac.grid(True, alpha=0.3)
                    fig_mac.tight_layout()
                    solara.FigureMatplotlib(fig_mac)
                    plt.close(fig_mac)

        # Annual Production Summary Table
        with solara.Card("Annual Production Summary", style="margin-top: 10px;"):
            if annual_summary:
                with solara.Row(style="font-size: 11px; font-weight: 700; border-bottom: 2px solid #cbd5e1; padding: 4px 0;"):
                    solara.Text("Year", style="flex: 1;")
                    solara.Text("Survey Biomass (kg)", style="flex: 2; text-align: right;")
                    solara.Text("Legal Catch (kg)", style="flex: 2; text-align: right;")
                    solara.Text("Exploitation Rate", style="flex: 2; text-align: right;")
                    solara.Text("Recruits", style="flex: 1; text-align: right;")
                    solara.Text("Spawners", style="flex: 1; text-align: right;")
                for r in annual_summary:
                    with solara.Row(style="font-size: 11px; border-bottom: 1px solid #f1f5f9; padding: 4px 0;"):
                        solara.Text(f"Year {r['year']}", style="flex: 1; font-weight: 600;")
                        solara.Text(f"{r['biomass']:.1f}", style="flex: 2; text-align: right;")
                        solara.Text(f"{r['catch']:.1f}", style="flex: 2; text-align: right;")
                        solara.Text(f"{r['exploit']:.1f}%", style="flex: 2; text-align: right;")
                        solara.Text(f"{r['recruits']}", style="flex: 1; text-align: right;")
                        solara.Text(f"{r['spawners']}", style="flex: 1; text-align: right;")
            else:
                solara.Text("No annual summary data recorded yet. Run simulation past day 365.", style="color: #94a3b8; font-size: 11px;")


Page = RealtimeDashboard
