"""
PettingZoo Parallel wrapper for multi-agent ViZDoom.

- Spawns one process per agent (host + peers). Communication via duplex Pipes.
- Parent sends simple commands ("reset", "step", "close") to each child and
  receives observations, rewards, terminations, truncations, and infos.

Usage:
    env = VizdoomParallelEnv(
        scenario="health_gathering",
        num_agents=2,
        resolution="160X120",
        skip_frames=4,
        async_mode=True,
        host_address="127.0.0.1",
        port=5029,
        netmode=0,  # 0: p2p, 1: client-server
        seed=0,
    )

    obs, infos = env.reset()
    done = False
    while not any(env._terminations.values()) and not any(env._truncations.values()):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, infos = env.step(actions)
    env.close()
"""
from __future__ import annotations

import math
import multiprocessing as mp
import sys

ctx = mp.get_context("spawn")
import subprocess
import pickle
from pathlib import Path
import time

from pettingzoo import ParallelEnv
from pettingzoo_wrapper.utils import parse_hw, discover_buttons
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

import vizdoom as vzd
import pygame
import cv2


# -------------------------- main PettingZoo env ---------------------------

class VizdoomParallelEnv(ParallelEnv):

    def __init__(
            self,
            *,
            config_file: str,
            num_agents: int = 2,
            resolution: str = "160X120",
            timeout: Optional[int] = None,
            skip_frames: Optional[int] = 1,
            async_mode: bool = True,
            host_address: str = "127.0.0.1",
            port: int = 5029,
            netmode: int = 0,
            ticrate: int = vzd.DEFAULT_TICRATE,
            render_mode: Optional[str] = None,
            use_multi_binary_action_space: bool = False,
            simple_discrete: bool = True,
            seed: Optional[int] = None,
            verbose: bool = False,
    ) -> None:
        assert num_agents >= 1
        self.config_file = config_file
        self._num_agents = num_agents
        self.host_address = host_address
        self.port = int(port)
        self.resolution = resolution
        self.netmode = int(netmode)
        self.async_mode = bool(async_mode)
        self.ticrate = int(ticrate)
        self.render_mode = render_mode
        self.use_multi_binary_action_space = bool(use_multi_binary_action_space)
        self.simple_discrete = bool(simple_discrete)
        self._ext_seed = seed

        # names
        self.possible_agents: List[str] = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents: List[str] = self.possible_agents[:]

        # Discover spaces (no net init needed)
        self._delta_count, self._binary_count = discover_buttons(config_file)
        self._simple_n = (3 ** self._delta_count) * (2 ** self._binary_count)
        self._act_len = self._delta_count + self._binary_count
        self._action_space = self._build_action_space()

        w, h = parse_hw(resolution)
        self._channels = 3  # will update on first reset if GRAY8
        self._obs_shape = (h, w, self._channels)
        self._observation_space = spaces.Box(0, 255, shape=self._obs_shape, dtype=np.uint8)

        # Child processes and pipes
        self._pipes: Dict[str, Tuple[Any, Any]] = {}  # agent_name -> (stdin, stdout)
        self._procs: List[subprocess.Popen] = []

        # Find the absolute path to the worker script
        worker_path = Path(__file__).parent / "vizdoom_worker.py"
        if not worker_path.is_file():
            raise FileNotFoundError(
                f"The 'vizdoom_worker.py' script was not found. "
                f"Please ensure it is in the same directory as this file: {worker_path.parent}"
            )

        for i in range(self._num_agents):
            agent_name = self.possible_agents[i]

            # 1. Package all arguments for the worker process
            worker_kwargs = dict(
                config_path=self.config_file, resolution=self.resolution, timeout=timeout, skip_frames=skip_frames,
                num_agents=self._num_agents, agent_idx=i, is_host=(i == 0), host_address=self.host_address,
                port=self.port, async_mode=self.async_mode, netmode=self.netmode, ticrate=self.ticrate,
                seed=(None if seed is None else int(seed) + i), verbose=verbose,
            )

            # 2. Serialize arguments to pass them on the command line
            hex_kwargs = pickle.dumps(worker_kwargs).hex()

            # 3. Launch the worker script as an independent subprocess
            cmd = [sys.executable, str(worker_path), hex_kwargs]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

            self._procs.append(proc)
            self._pipes[agent_name] = (proc.stdin, proc.stdout)

        # timeout / PZ bookkeeping
        self._frames_advanced = 0
        self._timeout = int(timeout) if timeout is not None else None

        # Death tracking for respawn functionality
        self._dead_agents = set()  # Track which agents are currently dead
        self._terminations: Dict[str, bool] = {a: False for a in self.agents}
        self._truncations: Dict[str, bool] = {a: False for a in self.agents}

        # last frames for rendering
        self._last_frames: Dict[str, np.ndarray] = {}

        # Rendering surface
        self._screen: Optional[pygame.Surface] = None

        # Give children a moment to init networking
        time.sleep(1.0)

    # ------------- space helpers -------------

    def _build_action_space(self) -> spaces.Space:
        if self.simple_discrete:
            return spaces.Discrete(max(1, self._simple_n))
        if self._delta_count == 0:
            return self._binary_space()
        if self._binary_count == 0:
            return self._continuous_space()
        return spaces.Dict({
            "binary": self._binary_space(),
            "continuous": self._continuous_space(),
        })

    def _binary_space(self) -> spaces.Space:
        if self.use_multi_binary_action_space:
            return spaces.MultiBinary(self._binary_count)
        return spaces.MultiDiscrete([2] * self._binary_count)

    def _continuous_space(self) -> spaces.Space:
        low, high = np.finfo(np.float32).min, np.finfo(np.float32).max
        return spaces.Box(low, high, (self._delta_count,), dtype=np.float32)

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_space

    @property
    def num_agents(self) -> int:
        return self._num_agents

    # ------------- PZ API -------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._ext_seed = int(seed)
        self._frames_advanced = 0
        self._terminations = {a: False for a in self.agents}
        self._truncations = {a: False for a in self.agents}

        # Clear dead agents tracking on reset
        self._dead_agents.clear()

        # broadcast reset, collect results
        for agent in self.agents:
            stdin, _ = self._pipes[agent]
            pickle.dump(("reset", None), stdin)
            stdin.flush()

        # Collect results from all workers
        results_dict = {agent: pickle.load(self._pipes[agent][1]) for agent in self.agents}

        obs: Dict[str, np.ndarray] = {}
        infos: Dict[str, Dict[str, Any]] = {}
        for i, agent in enumerate(self.agents):
            frame = results_dict[agent]["obs"]
            # update channel inference once
            c = frame.shape[2]
            if c != self._obs_shape[2]:
                self._obs_shape = (self._obs_shape[0], self._obs_shape[1], c)
                self._observation_space = spaces.Box(0, 255, shape=self._obs_shape, dtype=np.uint8)
            obs[agent] = frame
            infos[agent] = results_dict[agent].get("info", {})
            self._last_frames[agent] = frame

        return obs, infos

    def step(self, actions: Dict[str, Any]):
        # 1) Handle dead agents from previous step and encode actions for alive agents
        flat_actions: List[List[float]] = []
        for agent in self.agents:
            a = actions.get(agent, self._noop_action())
            env_action = self._encode_env_action(a)
            if len(env_action) != self._act_len:
                raise ValueError(f"Encoded action length {len(env_action)} != expected {self._act_len}")
            flat_actions.append(env_action)

        # Send commands: respawn for dead agents, step for alive agents
        for i, agent in enumerate(self.agents):
            cmd = "respawn" if agent in self._dead_agents else "step"
            stdin, _ = self._pipes[agent]
            pickle.dump((cmd, flat_actions[i]), stdin)
            stdin.flush()

        # Receive results from all workers
        results_dict = {agent: pickle.load(self._pipes[agent][1]) for agent in self.agents}

        observations: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        infos: Dict[str, Dict[str, Any]] = {}
        any_term = False

        # Track agents that were dead and got respawned this step
        respawned_agents = set()

        for i, agent in enumerate(self.agents):
            results_a = results_dict[agent]
            frame = results_a["obs"]
            observations[agent] = frame
            rewards[agent] = float(results_a.get("reward", 0.0))
            term = bool(results_a.get("terminated", False))
            self._terminations[agent] = term
            any_term = any_term or term
            infos[agent] = results_a.get("info", {})
            self._last_frames[agent] = frame

            # Handle respawn results for agents that were dead
            if agent in self._dead_agents:
                respawned = results_a.get("respawned", False)
                if respawned:
                    respawned_agents.add(agent)

        # Remove successfully respawned agents from dead set
        self._dead_agents -= respawned_agents

        # Track newly dead agents (but don't respawn them until next step)
        for i, agent in enumerate(self.agents):
            if infos[agent].get("player_dead", False) and agent not in self._dead_agents:
                self._dead_agents.add(agent)

        # 3) time-limit truncation (assume same num_frames for all)
        frames_advanced = next(iter(infos.values())).get("num_frames", 1) if infos else 1
        self._frames_advanced += int(frames_advanced)
        timeout_hit = (self._timeout is not None) and (self._frames_advanced >= self._timeout)
        if timeout_hit:
            for a in self.agents:
                self._truncations[a] = True
                infos[a]["TimeLimit.truncated"] = True

        if any(self._terminations.values()):
            for a in self.agents: self._terminations[a] = True
        if any(bool(r.get("truncated", False)) for r in results_dict.values()):
            for a in self.agents: self._truncations[a] = True

        return observations, rewards, self._terminations.copy(), self._truncations.copy(), infos

    def close(self):
        for agent in self.agents:
            try:
                stdin, _ = self._pipes.get(agent, (None, None))
                if stdin and not stdin.closed:
                    pickle.dump(("close", None), stdin)
                    stdin.flush()
                    stdin.close()
            except (IOError, BrokenPipeError):
                pass

        # Terminate and wait for the processes to exit
        for p in self._procs:
            try:
                p.terminate()  # Send SIGTERM
                p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                p.kill()  # Force kill if unresponsive
            except Exception:
                pass
        self._procs.clear()
        self._pipes.clear()
        # pygame
        if self._screen is not None:
            try:
                pygame.quit()
            except Exception:
                pass
        self._screen = None

    # ------------- helpers -------------
    def _noop_action(self):
        """Build a valid 'do nothing' action matching self._action_space."""
        if isinstance(self._action_space, spaces.Dict):
            out = {}
            if self._delta_count:
                out["continuous"] = np.zeros((self._delta_count,), dtype=np.float32)
            if self._binary_count:
                if self.use_multi_binary_action_space:
                    out["binary"] = np.zeros((self._binary_count,), dtype=np.int8)
                else:
                    out["binary"] = np.zeros((self._binary_count,), dtype=np.int64)
            return out
        else:
            if isinstance(self._action_space, spaces.Discrete):
                return 0
            if isinstance(self._action_space, spaces.MultiBinary):
                return np.zeros((self._binary_count,), dtype=np.int8)
            if isinstance(self._action_space, spaces.MultiDiscrete):
                return np.zeros((self._binary_count,), dtype=np.int64)
            if isinstance(self._action_space, spaces.Box):
                return np.zeros((self._delta_count,), dtype=np.float32)
            raise NotImplementedError(type(self._action_space))

    def _decode_simple_discrete(self, idx: int) -> List[float]:
        """Decode Discrete index -> flat [delta..., binary...] for ViZDoom.

        delta in {-1, 0, +1} (radix-3), then binary in {0,1} (radix-2).
        Order is [delta_0, ..., delta_{D-1}, bin_0, ..., bin_{B-1}]
        """
        D, B = self._delta_count, self._binary_count
        out = np.zeros((self._act_len,), dtype=np.float32)
        x = int(idx)

        # Binary tail (radix-2)
        for i in range(B):
            out[D + i] = float(x & 1)
            x >>= 1

        # Delta head (radix-3) mapped {0,1,2} -> {-1,0,+1}
        for i in range(D):
            digit = x % 3
            out[i] = float([-1, 0, +1][digit])
            x //= 3

        return out.tolist()

    def _encode_env_action(self, agent_action: Any) -> List[float]:
        if self.simple_discrete:
            # agent_action is an integer index from spaces.Discrete
            return self._decode_simple_discrete(int(agent_action))

        # Map user action (matching self._action_space) -> flat vector [delta..., binary...]
        out = np.zeros((self._act_len,), dtype=np.float32)
        if isinstance(self._action_space, spaces.Dict):
            # Dict with continuous and binary
            if self._delta_count:
                cont = agent_action["continuous"] if isinstance(agent_action, dict) else agent_action[0]
                out[: self._delta_count] = np.asarray(cont, dtype=np.float32)
            if self._binary_count:
                bin_act = agent_action["binary"] if isinstance(agent_action, dict) else agent_action[1]
                bin_arr = np.asarray(bin_act, dtype=np.float32)
                out[self._delta_count:] = bin_arr.reshape(-1)
        else:
            if self._delta_count:
                out[: self._delta_count] = np.asarray(agent_action, dtype=np.float32)
            else:
                out[self._delta_count:] = np.asarray(agent_action, dtype=np.float32)
        return out.tolist()

    # ------------------- rendering -------------------
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None or not self._last_frames:
            return None
        frames = [self._last_frames[a] for a in self.agents if a in self._last_frames]
        if not frames:
            return None
        h, w = frames[0].shape[:2]
        cols = min(self._num_agents, max(1, 1920 // w))
        rows = math.ceil(self._num_agents / cols)
        total_w = cols * w
        total_h = rows * h
        max_w, max_h = 1920, 1080
        scale = min(max_w / total_w if total_w > max_w else 1.0, max_h / total_h if total_h > max_h else 1.0)
        sw, sh = int(w * scale), int(h * scale)
        disp_w, disp_h = min(cols * sw, max_w), min(rows * sh, max_h)

        if self.render_mode == "human":
            if self._screen is None or self._screen.get_size() != (disp_w, disp_h):
                pygame.init()
                self._screen = pygame.display.set_mode((disp_w, disp_h))
                pygame.display.set_caption("ViZDoom Multi-Agent")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            for i, frame in enumerate(frames):
                if scale < 1.0:
                    frame = cv2.resize(frame, (sw, sh))
                col = i % cols
                row = i // cols
                x, y = col * sw, row * sh
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self._screen.blit(surf, (x, y))
            pygame.display.flip()
            time.sleep(0.01)
            return None
        else:  # rgb_array
            ch = frames[0].shape[2]
            canvas = np.zeros((disp_h, disp_w, ch), dtype=frames[0].dtype)
            for i, frame in enumerate(frames):
                if scale < 1.0:
                    frame = cv2.resize(frame, (sw, sh))
                col = i % cols
                row = i // cols
                x, y = col * sw, row * sh
                canvas[y: y + sh, x: x + sw] = frame[: sh, : sw]
            return canvas
