"""Solara dashboard for AgentCrab."""

from __future__ import annotations

import solara

from agentcrab.simulation import CrabModel


@solara.component
def Dashboard():
    n_agents = solara.use_reactive(100)
    days = solara.use_reactive(30)
    seed = solara.use_reactive(42)

    model = solara.use_memo(
        lambda: CrabModel(n_agents=n_agents.value, grid_size=(50, 50), seed=seed.value),
        dependencies=[n_agents.value, days.value, seed.value],
    )

    result = solara.use_memo(
        lambda: model.run(days.value, when=1),
        dependencies=[model],
    )

    with solara.Card("AgentCrab Dashboard"):
        with solara.Row():
            solara.SliderInt("Agents", value=n_agents, min=10, max=500)
            solara.SliderInt("Days", value=days, min=1, max=365)
            solara.SliderInt("Seed", value=seed, min=1, max=9999)

        if result is not None:
            model_df, agent_df = result
            solara.DataFrame(model_df.tail(10))

    return


def main():
    solara.run(Dashboard)


if __name__ == "__main__":
    main()
