# AgentCrab

Python (Mesa) port of TethysBio — an individual-based simulation engine for biological populations.

## Quick Start

```bash
cd agentcrab
pip install -e .
python -m agentcrab.cli --config configs/default.toml --days 365
```

## Structure

See `agentcrab_plan.md` for the full porting plan.

## Status

Phase 1 (Core Biology) and Phase 2 (ABM Engine) implemented. Remaining phases in progress.


# agentcrab (Python / Mesa / Solara)

cd c:/home/jae/projects/agentcrab

.\.venv\Scripts\python -m agentcrab.cli --config configs/snowcrab.toml --days 365 --n_agents 500 --grid 100 100 

.\.venv\Scripts\solara run src/agentcrab/realtime_dashboard.py

Start-Process index.html
