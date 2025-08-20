"""Parallel PettingZoo-style wrapper for cooperative multi-agent VizDoom."""

from __future__ import annotations

import itertools
import warnings
from typing import List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from pettingzoo.utils.env import ParallelEnv
except Exception:  # pragma: no cover
    class ParallelEnv:  # type: ignore
        """Fallback stub when PettingZoo is not installed."""
        pass

import vizdoom.vizdoom as vzd


class VizdoomParallelEnv(ParallelEnv):
    """A simple cooperative multi-agent wrapper around :class:`vizdoom.DoomGame`.

    The environment exposes a PettingZoo ``ParallelEnv``-style API where
    ``step`` expects a list of actions (one per agent) and returns lists of
    observations, rewards, terminations and truncations. The underlying
    ``DoomGame`` instances are kept in sync such that every agent observes the
    same game tick.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": vzd.DEFAULT_TICRATE}

    def __init__(
        self,
        config_file: str,
        num_agents: int,
        frame_skip: int = 1,
        max_buttons_pressed: int = 0,
        render_mode: Optional[str] = None,
        treat_episode_timeout_as_truncation: bool = True,
        use_multi_binary_action_space: bool = True,
        port: int = 50300,
    ) -> None:
        self.frame_skip = frame_skip
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.treat_episode_timeout_as_truncation = treat_episode_timeout_as_truncation
        self.use_multi_binary_action_space = use_multi_binary_action_space

        # create games
        self.games: List[vzd.DoomGame] = []
        for idx in range(num_agents):
            game = vzd.DoomGame()
            game.load_config(config_file)
            game.set_window_visible(False)
            if idx == 0:
                game.add_game_args(f"-host {num_agents} -port {port}")
            else:
                game.add_game_args(f"-join 127.0.0.1:{port}")
            game.set_mode(vzd.Mode.PLAYER)
            self.games.append(game)

        # use first game to infer observation/action spaces
        self.game = self.games[0]
        screen_format = self.game.get_screen_format()
        if screen_format not in (vzd.ScreenFormat.RGB24, vzd.ScreenFormat.GRAY8):
            warnings.warn(
                f"Detected screen format {screen_format.name}. Only RGB24 and GRAY8 are supported in the PettingZoo wrapper."
                " Forcing RGB24."
            )
            for g in self.games:
                g.set_screen_format(vzd.ScreenFormat.RGB24)
            screen_format = vzd.ScreenFormat.RGB24

        self.channels = 1 if screen_format == vzd.ScreenFormat.GRAY8 else 3
        self.depth = self.game.is_depth_buffer_enabled()
        self.labels = self.game.is_labels_buffer_enabled()
        self.automap = self.game.is_automap_buffer_enabled()

        self.__parse_available_buttons()

        if max_buttons_pressed > self.num_binary_buttons > 0:
            warnings.warn(
                f"max_buttons_pressed={max_buttons_pressed} > number of binary buttons defined={self.num_binary_buttons}."
                f" Clipping max_buttons_pressed to {self.num_binary_buttons}."
            )
            max_buttons_pressed = self.num_binary_buttons
        elif max_buttons_pressed < 0:
            raise RuntimeError("max_buttons_pressed must be >= 0")

        self.max_buttons_pressed = max_buttons_pressed
        self.action_space = self.__get_action_space()
        self.observation_space = self.__get_observation_space()

        for game in self.games:
            game.init()
        self.states = [None] * self.num_agents

    # ------------------------------------------------------------------
    # helper methods borrowed from the Gymnasium wrapper
    def __parse_available_buttons(self) -> None:
        delta_buttons = []
        binary_buttons = []
        for button in self.game.get_available_buttons():
            if vzd.is_delta_button(button) and button not in delta_buttons:
                delta_buttons.append(button)
            else:
                binary_buttons.append(button)
        self.game.set_available_buttons(delta_buttons + binary_buttons)
        self.num_delta_buttons = len(delta_buttons)
        self.num_binary_buttons = len(binary_buttons)
        if self.num_delta_buttons == self.num_binary_buttons == 0:
            raise RuntimeError(
                "No game buttons defined. Must specify game buttons using `available_buttons` in the config file."
            )

    def __get_binary_action_space(self):
        import gymnasium as gym

        if self.max_buttons_pressed == 0:
            if self.use_multi_binary_action_space:
                button_space = gym.spaces.MultiBinary(self.num_binary_buttons)
            else:
                button_space = gym.spaces.MultiDiscrete([2] * self.num_binary_buttons)
        else:
            self.button_map = [
                np.array(list(action))
                for action in itertools.product((0, 1), repeat=self.num_binary_buttons)
                if self.max_buttons_pressed >= sum(action) >= 0
            ]
            button_space = gym.spaces.Discrete(len(self.button_map))
        return button_space

    def __get_continuous_action_space(self):
        import gymnasium as gym

        return gym.spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            (self.num_delta_buttons,),
            dtype=np.float32,
        )

    def __get_action_space(self):
        import gymnasium as gym

        if self.num_delta_buttons == 0:
            return self.__get_binary_action_space()
        elif self.num_binary_buttons == 0:
            return self.__get_continuous_action_space()
        else:
            return gym.spaces.Dict(
                {
                    "binary": self.__get_binary_action_space(),
                    "continuous": self.__get_continuous_action_space(),
                }
            )

    def __get_observation_space(self):
        import gymnasium as gym

        spaces = {
            "screen": gym.spaces.Box(
                0,
                255,
                (self.game.get_screen_height(), self.game.get_screen_width(), self.channels),
                dtype=np.uint8,
            )
        }
        if self.depth:
            spaces["depth"] = gym.spaces.Box(
                0,
                255,
                (self.game.get_screen_height(), self.game.get_screen_width(), 1),
                dtype=np.uint8,
            )
        if self.labels:
            spaces["labels"] = gym.spaces.Box(
                0,
                255,
                (self.game.get_screen_height(), self.game.get_screen_width(), 1),
                dtype=np.uint8,
            )
        if self.automap:
            spaces["automap"] = gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    self.channels,
                ),
                dtype=np.uint8,
            )
        self.num_game_variables = self.game.get_available_game_variables_size()
        if self.num_game_variables > 0:
            spaces["gamevariables"] = gym.spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (self.num_game_variables,),
                dtype=np.float32,
            )
        return gym.spaces.Dict(spaces)

    def __build_env_action(self, agent_action):
        env_action = np.zeros(self.num_delta_buttons + self.num_binary_buttons, dtype=np.float32)
        self.__parse_delta_buttons(env_action, agent_action)
        self.__parse_binary_buttons(env_action, agent_action)
        return env_action

    def __parse_binary_buttons(self, env_action, agent_action):
        if self.num_binary_buttons != 0:
            if self.num_delta_buttons != 0:
                agent_action = agent_action["binary"]
            if np.issubdtype(type(agent_action), np.integer):
                agent_action = self.button_map[agent_action]
            env_action[self.num_delta_buttons :] = agent_action

    def __parse_delta_buttons(self, env_action, agent_action):
        if self.num_delta_buttons != 0:
            if self.num_binary_buttons != 0:
                agent_action = agent_action["continuous"]
            env_action[0 : self.num_delta_buttons] = agent_action

    def __collect_observations(self, idx: int):
        observation = {}
        state = self.states[idx]
        if state is not None:
            observation["screen"] = state.screen_buffer
            if self.channels == 1:
                observation["screen"] = state.screen_buffer[..., None]
            if self.depth:
                observation["depth"] = state.depth_buffer[..., None]
            if self.labels:
                observation["labels"] = state.labels_buffer[..., None]
            if self.automap:
                observation["automap"] = state.automap_buffer
                if self.channels == 1:
                    observation["automap"] = state.automap_buffer[..., None]
            if self.num_game_variables > 0:
                observation["gamevariables"] = state.game_variables.astype(np.float32)
        else:
            for space_key, space_item in self.observation_space.spaces.items():
                observation[space_key] = np.zeros(space_item.shape, dtype=space_item.dtype)
        return observation

    # ------------------------------------------------------------------
    # API methods
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            for game in self.games:
                game.set_seed(int(rng.integers(0, np.iinfo(np.uint32).max + 1)))
        for game in self.games:
            game.new_episode()
        self.states = [game.get_state() for game in self.games]
        observations = [self.__collect_observations(i) for i in range(self.num_agents)]
        return observations, {}

    def step(self, actions: Sequence):
        assert len(actions) == self.num_agents, "Number of actions must match number of agents"
        for game, act in zip(self.games, actions):
            env_action = self.__build_env_action(act)
            game.set_action(env_action)
        rewards = []
        terminations = []
        truncations = []
        infos = [{} for _ in range(self.num_agents)]
        observations = []
        for i, game in enumerate(self.games):
            reward = game.advance_action(self.frame_skip)
            self.states[i] = game.get_state()
            terminated = game.is_episode_finished()
            truncated = (
                game.is_episode_timeout_reached()
                if self.treat_episode_timeout_as_truncation
                else False
            )
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            observations.append(self.__collect_observations(i))
        return observations, rewards, terminations, truncations, infos

    def close(self):
        for game in self.games:
            game.close()
