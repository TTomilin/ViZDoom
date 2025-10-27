from pathlib import Path
import pickle
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from typing import Optional

import vizdoom as vzd
from vizdoom import Mode

from pettingzoo_wrapper.utils import screen_res, parse_hw, get_flat_game_vars, read_frame

def run_worker(
        *,
        config_path: str,
        resolution: str,
        timeout: int,
        skip_frames: Optional[int],
        num_agents: int,
        agent_idx: int,
        is_host: bool,
        host_address: str,
        port: int,
        async_mode: bool,
        netmode: int,
        ticrate: int,
        seed: Optional[int],
        verbose: bool,
) -> None:
    """
    Main execution loop for a single ViZDoom agent.
    It communicates via standard input/output using pickled Python objects.
    """
    # Use stdin/stdout for communication with the parent process
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    game = vzd.DoomGame()
    game.load_config(config_path)

    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_console_enabled(False)
    game.set_render_hud(True)
    game.set_screen_resolution(screen_res(resolution))
    game.set_ticrate(ticrate)
    game.set_mode(Mode.ASYNC_PLAYER if async_mode else Mode.PLAYER)

    if timeout is not None:
        game.set_episode_timeout(timeout)
    if seed is not None:
        game.set_seed(int(seed))

    # Network setup
    if is_host:
        game.add_game_args(
            f"-host {num_agents} -port {port} -netmode {netmode} +sv_spawnfarthest 1"
        )
        agent_name = "host"
    else:
        game.add_game_args(f"-join {host_address} -port {port} -netmode {netmode}")
        agent_name = f"peer{agent_idx}"

    game.add_game_args(f"+name Player{agent_idx} +colorset {agent_idx} +playernumber {agent_idx}")
    game.init()
    game.send_game_command("viz_respawn_delay 0")

    available_game_vars = game.get_available_game_variables()
    steps = 0
    is_dead = False
    frames_per_step = skip_frames if skip_frames else 1

    try:
        while True:
            # Blocking read from parent process
            cmd, payload = pickle.load(stdin)

            if cmd == "reset":
                game.new_episode()
                game.respawn_player()
                state = game.get_state()
                frame = read_frame(state, resolution)
                info = {"num_frames": frames_per_step, "player_died": False, "just_died": False, "step": steps}
                info.update(get_flat_game_vars(state, available_game_vars))
                result = {"obs": frame, "reward": 0.0, "terminated": False, "info": info}

            elif cmd == "step":
                reward = float(game.make_action(payload, skip_frames) if skip_frames else game.make_action(payload))
                was_dead_before = is_dead
                is_dead = game.is_player_dead()
                just_died = not was_dead_before and is_dead
                truncated = game.is_episode_finished()
                terminated = game.is_episode_finished()
                if verbose and terminated:
                    print(f"Player {agent_name} terminated at step {game.get_episode_time()}")
                state = game.get_state()
                frame = read_frame(state, resolution)
                info = {"num_frames": frames_per_step, "player_died": is_dead, "just_died": just_died, "step": steps}
                info.update(get_flat_game_vars(state, available_game_vars))
                result = {"obs": frame, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}
                steps += frames_per_step

            elif cmd == "respawn":
                if is_dead:
                    if verbose: print(f"Player {agent_name} respawning at step {game.get_episode_time()}...")
                    game.respawn_player()
                    is_dead = False
                    if verbose: print(f"Player {agent_name} respawned at step {game.get_episode_time()}")
                    respawned = True
                else:
                    zero_action = [0.0] * len(game.get_available_buttons())
                    game.make_action(zero_action)
                    respawned = False

                state = game.get_state()
                frame = read_frame(state, resolution)
                info = {"num_frames": frames_per_step, "player_died": is_dead, "just_died": False, "step": steps}
                info.update(get_flat_game_vars(state, available_game_vars))
                result = {"obs": frame, "reward": 0.0, "terminated": False, "truncated": game.is_episode_finished(), "info": info, "respawned": respawned}
                steps += frames_per_step

            elif cmd == "close":
                break
            else:
                # Send a dummy response for safety
                dummy_frame = read_frame(None, resolution)
                result = {"obs": dummy_frame, "reward": 0.0, "terminated": False, "info": {}}

            # Send the result back to the parent and flush the buffer
            pickle.dump(result, stdout)
            stdout.flush()
    finally:
        game.close()


if __name__ == "__main__":
    # This block allows the script to be executed directly.
    # It decodes the configuration passed as a command-line argument
    # and starts the worker function.
    try:
        hex_kwargs = sys.argv[1]
        kwargs = pickle.loads(bytes.fromhex(hex_kwargs))
        run_worker(**kwargs)
    except Exception as e:
        # If something goes wrong, write the error to a log file for debugging.
        with open("vizdoom_worker_error.log", "a") as f:
            f.write(f"--- ERROR ---\n")
            f.write(f"{e}\n")
            import traceback
            traceback.print_exc(file=f)
        sys.exit(1)