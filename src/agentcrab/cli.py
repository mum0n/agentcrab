"""Command-line interface for AgentCrab."""

from __future__ import annotations

import argparse
import sys

from .simulation import CrabModel
from .types import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AgentCrab IBM simulation")
    parser.add_argument("--assimilate", action="store_true", help="Download and cache environmental data")
    parser.add_argument("--config", default="configs/default.toml", help="TOML config file")
    parser.add_argument("--days", type=int, default=365, help="Simulation duration (days)")
    parser.add_argument("--n_agents", type=int, default=2000, help="Initial population size")
    parser.add_argument("--grid", type=int, nargs=2, default=[100, 100], help="Grid dimensions")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--save", action="store_true", help="Save results to Parquet")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots after simulation")
    parser.add_argument("--output", default="results", help="Output directory")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.assimilate:
        from .assimilation import assimilate_environmental_inputs
        status = assimilate_environmental_inputs()
        for k, v in status.items():
            print(f"  {k}: {v}")
        return

    model = CrabModel(
        n_agents=args.n_agents,
        grid_size=tuple(args.grid),
        seed=args.seed,
        config_path=args.config,
    )

    model_df, agent_df = model.run(args.days, when=7)

    print(f"Simulation complete: {args.days} days")
    print(f"Final population: {model.summarize()['n_agents']}")
    print(f"Mean carapace width: {model.summarize()['mean_cw']:.2f} mm")

    if args.save or args.plot:
        import os
        os.makedirs(args.output, exist_ok=True)
        model_df.to_parquet(f"{args.output}/model_results.parquet")
        agent_df.to_parquet(f"{args.output}/agent_results.parquet")
        print(f"Results saved to {args.output}/")

    if args.plot:
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
        from plot_results import main as plot_main
        plot_main(["--input", args.output, "--output", args.output])


if __name__ == "__main__":
    main()
