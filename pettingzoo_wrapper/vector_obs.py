"""
Generic "hidden state" vector observations.

By default agents only see pixels, so e.g. in health_gathering_multi_agent the
only way to know another agent's health is to infer it from their on-screen
colour. This module lets a scenario expose game variables directly to the
policy as a small vector.

Extending to a new scenario or new hidden parameters:
    Add an entry to SCENARIO_VECTOR_OBS mapping the scenario name to a
    VectorObsSpec listing the info/game-variable keys to expose (anything
    present in the env's per-step info dict, e.g. "HEALTH", "ARMOR",
    "AMMO2", "DEAD"). Nothing else needs to change. The wrapper builds
    the vector generically for however many agents and keys are configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseParallelWrapper


@dataclass(frozen=True)
class VectorObsSpec:
    """
    keys: game/info variables to expose, in this order, for every agent counted.
    scale: per-key normalization divisor (value / scale), applied before the
        vector reaches the network. Keys not listed default to 1.0 (no scaling).
    include_self: include the acting agent's own values.
    include_others: include the other agents' values, ordered by
        possible_agents (self excluded). This is what lets an agent "see"
        e.g. a teammate's health directly.
    """

    keys: Sequence[str]
    scale: Dict[str, float] = field(default_factory=dict)
    include_self: bool = True
    include_others: bool = True

    def vector_size(self, num_agents: int) -> int:
        agents_counted = (1 if self.include_self else 0) + (
            (num_agents - 1) if self.include_others else 0
        )
        return len(self.keys) * agents_counted

    def agent_order(self, agent: str, possible_agents: Sequence[str]) -> List[str]:
        order: List[str] = []
        if self.include_self:
            order.append(agent)
        if self.include_others:
            order.extend(a for a in possible_agents if a != agent)
        return order


# Per-scenario registry.
SCENARIO_VECTOR_OBS: Dict[str, VectorObsSpec] = {
    "health_gathering_multi_agent": VectorObsSpec(
        keys=("HEALTH",),
        scale={"HEALTH": 100.0},
    ),
}


class VectorStateObservationWrapper(BaseParallelWrapper):
    """
    Turns each agent's plain image observation into
    {"image": <image>, "vector": <float32 vector>}, where the vector is built
    from `spec` out of that step's info dict.

    Must wrap the raw (image-observation) env directly. Anything downstream
    that expects a flat image array (e.g. VideoLoggerParallelWrapper) should
    be *between* the raw env and this wrapper, not outside it.
    """

    def __init__(self, env, spec: VectorObsSpec):
        super().__init__(env)
        self.spec = spec
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self._vector_size = spec.vector_size(len(self.possible_agents))

    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Dict(
            {
                "image": self.env.observation_space(agent),
                "vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._vector_size,),
                    dtype=np.float32,
                ),
            }
        )

    def _build_vector(self, agent: str, infos: Dict[str, Dict[str, Any]]) -> np.ndarray:
        spec = self.spec
        values: List[float] = []
        for other in spec.agent_order(agent, self.possible_agents):
            other_info = infos.get(other) or {}
            reset_info = other_info.get("reset_info")
            source = reset_info if isinstance(reset_info, dict) else other_info
            for key in spec.keys:
                raw = source.get(key, 0.0)
                values.append(float(raw) / spec.scale.get(key, 1.0))
        return np.asarray(values, dtype=np.float32)

    def _augment(
        self,
        obs: Dict[str, np.ndarray],
        infos: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            agent: {"image": image, "vector": self._build_vector(agent, infos)}
            for agent, image in obs.items()
        }

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        return self._augment(obs, infos), infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = self.env.agents[:]
        return self._augment(obs, infos), rewards, terminations, truncations, infos
