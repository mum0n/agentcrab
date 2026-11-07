"""Run a simulation and generate visualization plots."""

import os
import sys

sys.path.insert(0, "src")

from agentcrab.simulation import CrabModel
from agentcrab.visualization import plot_simulation_dashboard, plot_population_timeseries

# Run simulation
model = CrabModel(n_agents=200, grid_size=(50, 50), seed=42)
model_df, agent_df = model.run(100, when=1)

# Generate dashboard
os.makedirs("output", exist_ok=True)
fig = plot_simulation_dashboard(model, model_df, agent_df, save_path="output/dashboard.png")
print(f"Dashboard saved to output/dashboard.png")
print(f"Final population: {model.summarize()['n_agents']}")
print(f"Mean CW: {model.summarize()['mean_cw']:.2f} mm")

# Also save timeseries
fig2 = plot_population_timeseries(model_df, save_path="output/timeseries.png")
print(f"Timeseries saved to output/timeseries.png")
