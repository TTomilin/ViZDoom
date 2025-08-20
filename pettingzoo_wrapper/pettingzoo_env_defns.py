import os

from vizdoom import scenarios_path
from vizdoom.pettingzoo_wrapper.base_pettingzoo_env import VizdoomParallelEnv


class VizdoomParallelScenarioEnv(VizdoomParallelEnv):
    """Multi-agent environments for scenarios packaged with ViZDoom."""

    def __init__(
        self,
        scenario_config_file: str,
        num_agents: int,
        frame_skip: int = 1,
        max_buttons_pressed: int = 0,
        render_mode: str | None = None,
        treat_episode_timeout_as_truncation: bool = True,
        use_multi_binary_action_space: bool = True,
    ) -> None:
        super().__init__(
            os.path.join(scenarios_path, scenario_config_file),
            num_agents,
            frame_skip,
            max_buttons_pressed,
            render_mode,
            treat_episode_timeout_as_truncation,
            use_multi_binary_action_space,
        )
