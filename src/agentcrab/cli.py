"""Command-line interface for AgentCrab."""

from __future__ import annotations

import argparse
import sys

from .simulation import CrabModel
from .types import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AgentCrab IBM simulation (TethysBio Python port)")
    parser.add_argument("--assimilate", action="store_true", help="Assimilate environmental data and write manifest")
    parser.add_argument("--config", default="configs/snowcrab.toml", help="TOML config file")
    parser.add_argument("--days", type=int, default=365, help="Simulation duration (days)")
    parser.add_argument("--dt", type=float, default=None, help="Timestep duration (days)")
    parser.add_argument("--frequency_days", type=int, default=7, help="Collection cadence (days)")
    parser.add_argument("--n_agents", type=int, default=2000, help="Initial population size")
    parser.add_argument("--grid", type=int, nargs=2, default=None, help="Grid dimensions (nx ny)")
    parser.add_argument("--grid_nx", type=int, default=None, help="Grid X cells")
    parser.add_argument("--grid_ny", type=int, default=None, help="Grid Y cells")
    parser.add_argument("--temp_offset", type=float, default=None, help="Temperature offset (C)")
    parser.add_argument("--temp_amplitude", type=float, default=None, help="Seasonal temperature amplitude (C)")
    parser.add_argument("--benthic_carrying_capacity", type=float, default=None, help="Carrying capacity")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--save", action="store_true", help="Save results to Parquet lakehouse files")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots after simulation")
    parser.add_argument("--html", action="store_true", help="Open standalone HTML dashboard")
    parser.add_argument("--gui", action="store_true", help="Launch Solara realtime dashboard")
    parser.add_argument("--output", default="results", help="Output directory")
    args, unknown = parser.parse_known_args(argv)
    args._overrides = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if tok.startswith("--"):
            key = tok[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                args._overrides[k] = v
                i += 1
            elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                args._overrides[key] = unknown[i + 1]
                i += 2
            else:
                args._overrides[key] = True
                i += 1
        else:
            i += 1
    return args


def _coerce_cli_val(val):
    if isinstance(val, bool):
        return val
    s = str(val)
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def print_simulation_report(model: CrabModel, days: int) -> None:
    summ = model.summarize()
    print(f"Simulation complete: {days} days")
    print(f"  Final population: {summ['n_agents']}")
    print(f"  Mean carapace width: {summ['mean_cw']:.2f} mm")
    print(f"  Biomass (kg): {summ['biomass_kg']:.2f}")
    print(f"  Mature: {summ['n_mature']}, Legal males: {summ['n_legal']}")
    print(f"  Catch count: {summ['catch_total']}, Catch biomass (kg): {summ['catch_biomass_kg']:.2f}")


def main(argv=None):
    args = parse_args(argv)

    if args.assimilate:
        from .assimilation import assimilate_environmental_inputs

        status = assimilate_environmental_inputs()
        for k, v in status.items():
            print(f"  {k}: {v}")
        return

    if args.html:
        import webbrowser
        from pathlib import Path

        idx = Path(__file__).resolve().parents[2] / "index.html"
        if idx.exists():
            webbrowser.open(idx.as_uri())
            print(f"Opened {idx}")
        return

    if args.gui:
        import subprocess
        from pathlib import Path

        rt = Path(__file__).resolve().parent / "realtime_dashboard.py"
        subprocess.run([sys.executable, "-m", "solara", "run", str(rt)], check=False)
        return

    from .types import with_overrides
    from .analysis import save_simulation_results

    params = load_config(args.config)
    overrides = {k: _coerce_cli_val(v) for k, v in getattr(args, "_overrides", {}).items()}
    if args.dt is not None:
        overrides["dt"] = args.dt
    if args.temp_offset is not None:
        overrides["temp_offset"] = args.temp_offset
    if args.temp_amplitude is not None:
        overrides["temp_multiplier"] = args.temp_amplitude
    if args.benthic_carrying_capacity is not None:
        overrides["benthic_carrying_capacity"] = args.benthic_carrying_capacity
    if args.grid_nx is not None:
        overrides["grid_nx"] = args.grid_nx
    if args.grid_ny is not None:
        overrides["grid_ny"] = args.grid_ny
    if overrides:
        params = with_overrides(params, overrides)

    grid_sz = tuple(args.grid) if args.grid is not None else (params.simulation.grid_nx, params.simulation.grid_ny)
    model = CrabModel(
        n_agents=args.n_agents,
        grid_size=grid_sz,
        seed=args.seed,
        params=params,
    )

    cadence = max(1, int(round(args.frequency_days / max(0.1, model.dt))))
    n_steps = max(1, int(round(args.days / max(0.1, model.dt))))
    model_df, agent_df = model.run(n_steps, when=cadence)
    print_simulation_report(model, args.days)

    if args.save or args.plot:
        paths = save_simulation_results(model_df, agent_df, output_dir=args.output)
        print(f"Results saved to {args.output}/ ({', '.join(paths.keys())})")

    if args.plot:
        from .visualization import export_all_plots

        export_all_plots(model, model_df, agent_df, output_dir=args.output)


if __name__ == "__main__":
    main()
