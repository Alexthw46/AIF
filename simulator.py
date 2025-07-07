import itertools
import time

import IPython.display as display
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from minihack import LevelGenerator
from minihack import RewardManager
from nle import nethack
from tqdm import tqdm

import utils


def stairs_reward_function(env, previous_observation, action, observation):
    # Agent is on stairs down
    if observation[env._internal_index][4]:
        return 1
    return 0


def simulate_offline_planning(env, fun: callable, verbose: bool = True, clear_outputs: bool = True,
                              wait_time: float = 0.5,
                              cropped: bool = True, save_dir=None,
                              gif_name: str = '', **kwargs):
    """
    Simulates the static environment using a pathfinding function.

    Parameters:
    :param env: The environment to simulate.
    :param  fun: The pathfinding function to use for simulation. Need to follow the signature:
        fun(game_map, start, target, apple_positions, **kwargs)
    :param verbose: Whether to print detailed information during the simulation.
    :param clear_outputs: Whether to clear the output display after each action.
    :param wait_time: Time to wait between actions for visualization.
    :param cropped: Whether to use the cropped pixel observation.
    :param save_dir: Directory to save the simulation images as a video.
    :param gif_name: Name of the video file to save.

    :param  kwargs: Additional keyword arguments for the heuristic function.

    Returns:
    - The result of the simulation.
    """

    state, _ = env.reset()
    game_map = state['chars']
    game = state['pixel_crop' if cropped else 'pixel']
    start = utils.get_player_location(game_map)
    target = utils.get_stairs_location(game_map)

    char_map = utils.np.vectorize(chr)(game_map)
    apple_positions = utils.np.where(char_map == '%')
    dic = {}

    # zip into a list of int tuples (x, y)
    apple_positions = list(zip(apple_positions[0], apple_positions[1]))
    # save initial number of apples 
    initial_num_apples = len(apple_positions)

    #    print("Apple positions:", [(int(x), int(y)) for x, y in apple_positions])

    start_time = time.time()
    path = fun(game_map, start, target, set(apple_positions), **kwargs)
    end_time = time.time()

    planning_time = end_time - start_time

    # save path length 
    path_length = len(path)

    if path is None or len(path) == 0:
        print("No path found.")
        return 0, -1, planning_time, initial_num_apples, False, dic
    elif verbose:
        print("Path found:")
        for (x, y) in path:
            print(x, y)

    actions = utils.actions_from_path(start, path[1:])
    if verbose:
        utils.simulate_path(path, game_map, actions)

        image = plt.imshow(game if cropped else game[:, 400:850])

        time.sleep(2)
    timer = time.time()
    tot_reward = 0
    done = False
    images = []

    # For each action in the path, take the action and update the state
    for action in actions:
        s, reward, done, _, dic = env.step(action)
        tot_reward += reward

        if not done:
            if verbose:
                time.sleep(wait_time)
                timer += wait_time  # exclude the wait time from the total time
                if clear_outputs:
                    display.clear_output(wait=True)
                image.set_data(s['pixel_crop'] if cropped else s['pixel'][:, 300:975])
                images.append(s['pixel_crop'] if cropped else s['pixel'][:, 300:975])
                print("Action taken:", utils.directions[action])

            # check and eat apple
            s, reward, _, _, _ = check_and_eat_apple(s, env, apple_positions, verbose=verbose)
            tot_reward += reward
            if verbose:
                display.display(plt.gcf())
                print("Reward:", tot_reward)
        else:
            if verbose:
                print(f"Episode finished:", dic)
                print("Reward:", tot_reward)
            break
    if not done:
        print("Stairs were not reached.")

    # final number of apples 
    num_remaining_apples = len(apple_positions)
    # num. of eaten apples 
    num_eaten_apples = initial_num_apples - num_remaining_apples

    if verbose:
        print("Number of eaten apples: ", num_eaten_apples)
        print("Path length: ", path_length)
        print(f"Planning time: {planning_time:.4f} seconds")
        print(f"Episode terminated with success: ", done)
        print(f"Total collected reward: ", tot_reward)

        print(f"Simulation completed in {time.time() - timer:.2f} seconds.")

    if verbose and save_dir is not None:
        # Save the images as a video
        utils.save_images_as_video(images, save_dir=save_dir, file_name=gif_name, fps=5)

    return tot_reward, path_length, planning_time, num_eaten_apples, done, dic


def simulate_online(env, fun: callable, clear_outputs=True, wait_time: float = 0.5, cropped=True, save_dir=None,
                    gif_name='', verbose=True, **kwargs):
    """
    Simulates the dynamic environment using a frontier search plus a pathfinding function.

    Parameters:
    :param env: The environment to simulate.
    :param fun: The pathfinding function to use for simulation. Must follow the signature:
        fun(game_map, start, **kwargs)
    :param verbose: Whether to print detailed information during the simulation.
    :param clear_outputs: Whether to clear the output display after each action.
    :param wait_time: Time to wait between actions for visualization.
    :param cropped: Whether to use the cropped pixel observation.
    :param save_dir: Directory to save the simulation images as a video.
    :param gif_name: Name of the video file to save.

    Returns:
    - tot_reward: Total reward collected.
    - total_path_length: Sum of all path lengths followed.
    - total_planning_time: Total time spent on planning.
    - total_apples_eaten: Number of apples eaten.
    - done: Whether the episode finished successfully.
    - dic: Additional info from the environment.
    """
    state, _ = env.reset()
    game_map = state['chars']
    game = state['pixel_crop' if cropped else 'pixel']

    if verbose:
        image = plt.imshow(game if cropped else game[:, 400:850])
        time.sleep(2)

    timer = time.time()
    planning_time_total = 0
    total_path_length = 0
    total_apples_eaten = 0
    done = False
    dic = {}
    s = state
    old_apple_positions = []
    images = []
    tot_reward = 0

    while not done:
        start = utils.get_player_location(game_map)

        if verbose:
            print("Evaluating path...")

        t0 = time.time()
        path = fun(game_map, start, verbose=verbose, **kwargs)
        t1 = time.time()
        planning_time_total += (t1 - t0)

        if path is None or len(path) == 0:
            if verbose:
                print("No path found.")
            break

        total_path_length += len(path)
        actions = utils.actions_from_path(start, path[1:])
        if verbose:
            utils.simulate_path(path, game_map, actions)
            time.sleep(wait_time)

        for action in actions:
            apple_positions = utils.np.where(utils.np.vectorize(chr)(s['chars']) == '%')
            apple_positions = list(zip(apple_positions[0], apple_positions[1]))

            # new apple discovered — stop and replan
            if len(apple_positions) > len(old_apple_positions):
                old_apple_positions = apple_positions
                if verbose:
                    print("Apple discovered, recomputing...")
                break

            s, reward, done, _, dic = env.step(action)
            tot_reward += reward

            if not done:
                if verbose:
                    time.sleep(wait_time)
                    timer += wait_time
                    if clear_outputs:
                        display.clear_output(wait=True)
                    print("Reward:", tot_reward)
                    image.set_data(s['pixel_crop'] if cropped else s['pixel'][:, 300:975])
                    images.append(s['pixel_crop'] if cropped else s['pixel'][:, 300:975])
                    print("Action taken:", utils.directions[action])
                    print(bytes(s['message']).decode('utf-8').rstrip('\x00'))

                # eat apple if on one
                s, reward, _, _, _ = check_and_eat_apple(s, env, apple_positions, verbose=verbose)
                tot_reward += reward
                total_apples_eaten += reward > 0  # assumes positive reward only when apple is eaten
            else:
                break

            old_apple_positions = apple_positions
            game_map = s['chars']

    if verbose:
        print(f"Episode finished: {dic}")
        print("Reward:", tot_reward)
        print(f"Simulation completed in {time.time() - timer:.2f} seconds.")

    if verbose and save_dir is not None:
        utils.save_images_as_video(images, save_dir=save_dir, file_name=gif_name, fps=5)

    return tot_reward, total_path_length, planning_time_total, total_apples_eaten, done, dic


def check_and_eat_apple(state, env, apple_positions, verbose=True):
    """
    Check if the player is on an apple and eat it.
    """

    PICKUP = env.unwrapped.actions.index(nethack.Command.PICKUP)
    EAT = env.unwrapped.actions.index(nethack.Command.EAT)
    # INVENTORY = env.unwrapped.actions.index(nethack.Command.INVENTORY)

    game_map = state['chars']
    player_location = utils.get_player_location(game_map)
    if verbose:
        print(f"Player location: {player_location}")
        print(f"Apple location: {apple_positions}")
    if player_location in apple_positions:
        if verbose: print("Found apple at:", player_location)
        # Pick up the apple
        s, _, done, _, dic = env.step(PICKUP)
        if verbose: print("ACTION_TAKEN: PICKUP")
        apple_positions.remove(player_location)  # Remove the apple from the list
        s, _, done, _, dic = env.step(EAT)
        # What do you want to eat?[g or *]
        msg = bytes(s['message']).decode('utf-8').rstrip('\x00')
        if verbose: print(msg)
        food_char = msg.split('[')[1][0]  # Because of the way the message in NetHack works

        # print("Food character to eat:", food_char)
        # Eat the apple
        # Open inventory to find the apple letter
        s, reward, done, boh, dic = env.step(env.unwrapped.actions.index(ord(food_char)))
        if verbose:
            print("ACTION_TAKEN: EAT", food_char)
            msg = bytes(s['message']).decode('utf-8').rstrip('\x00')
            print(msg)
        return s, reward, done, boh, dic

    # If not on an apple, just return the state
    return state, 0, False, None, None


def create_env(map, penalty_time: float = -0.1, apple_reward: float = 0.75) -> gym.Env:
    """
    Create a MiniHack environment with the specified map and reward settings.
    :param map: The map to use for the environment.
    :param penalty_time: The penalty time for each step taken.
    :param apple_reward: The reward for eating an apple.
    """

    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    ACTIONS = MOVE_ACTIONS + (nethack.Command.EAT, nethack.Command.PICKUP, nethack.Command.INVENTORY) + tuple(
        range(ord('a'), ord('z') + 1))

    # Define a reward manager
    reward_manager = RewardManager()
    # Reward for eating an apple
    reward_manager.add_eat_event("apple", reward=apple_reward, repeatable=True, terminal_required=False,
                                 terminal_sufficient=False)
    # Will never be achieved, but insures the environment keeps running until the stairs are reached
    reward_manager.add_message_event(
        ["Mission Complete."],
        terminal_required=True,
        terminal_sufficient=True,
    )
    # Add a custom reward function for stairs
    reward_manager.add_custom_reward_fn(stairs_reward_function)

    # Create the environment
    env = gym.make(
        "MiniHack-Skill-Custom-v0",
        des_file=map,
        reward_manager=reward_manager,
        observation_keys=("glyphs", "chars", "colors", 'screen_descriptions', 'inv_strs', 'blstats', 'message',
                          'pixel', 'pixel_crop'),
        actions=ACTIONS,
        penalty_time=penalty_time,
        obs_crop_h=15,
        obs_crop_w=15,
    )

    return env


def make_map(map_str: str, n_apples: int, seed=None, start=None, stairs=None, premapped=False) -> str:
    """
    Create a map file for the MiniHack environment.
    """
    lvl_gen = LevelGenerator(map=map_str, flags=("premapped" if premapped else "hardfloor",))
    cols, rows = lvl_gen.x, lvl_gen.y

    # Sync the map with the level generator post-initialization
    map_str = lvl_gen.get_map_str()

    if start is not None:
        lvl_gen.set_start_pos(start)
    # Randomly place the start position otherwise

    # Generate random positions for apples
    if seed is None:
        for pos in range(0, n_apples):
            lvl_gen.add_object("apple", "%")
    else:
        apple_positions = utils.randomize_apple_positions(map_str, 0, 1, cols, rows - 1, n_apples, seed=seed)

        for pos in apple_positions:
            lvl_gen.add_object("apple", "%", place=pos)

    # Randomly place the stairs down if no seed is provided
    if stairs is not None:
        lvl_gen.add_stair_down(stairs)
    elif seed is not None:
        lvl_gen.add_stair_down((cols - 2, rows - 2))
    else:
        lvl_gen.add_stair_down()

    lvl_gen.wallify()

    return lvl_gen.get_des()


def benchmark_simulation(env_fn, algorithm_fn, seeds, param_grid, online=False, **common_kwargs):
    """
    Benchmarks a pathfinding algorithm under different random seeds and parameter settings.

    Parameters:
    - env_fn: A callable to create the environment. Must accept a seed argument.
    - algorithm_fn: The pathfinding function to benchmark.
    - seeds: A list of seeds for environment randomization.
    - param_grid: A dict of parameter lists to explore (e.g., {'beam_width': [3, 5]}).
    - num_runs: Number of runs per combination of parameters and seed.
    - common_kwargs: Additional static keyword arguments for simulate_offline_planning.

    Returns:
    - A pandas DataFrame of the results.
    """

    results = []
    param_keys = list(param_grid.keys())
    param_combinations = list(itertools.product(*param_grid.values()))

    total = len(seeds) * len(param_combinations)
    pbar = tqdm(total=total, desc="Benchmarking")

    for seed in seeds:
        for param_values in param_combinations:
            param_dict = dict(zip(param_keys, param_values))

            env = env_fn(seed)

            try:
                result = (simulate_online if online else simulate_offline_planning)(
                    env,
                    algorithm_fn,
                    verbose=False,
                    wait_time=0.0,
                    **param_dict,
                    **common_kwargs
                )
                (tot_reward, path_len, plan_time, apples_eaten, success, episode_info) = result
            except Exception as e:
                print(f"[!] Failed: seed={seed}, params={param_dict} -> {e}")
                tot_reward, path_len, plan_time, apples_eaten, success, episode_info = [None] * 6

            result_row = {
                'seed': seed,
                **param_dict,
                'reward': tot_reward,
                'path_length': path_len,
                'planning_time': plan_time,
                'apples_eaten': apples_eaten,
                'success': success,
                **episode_info  # you can store custom flags from your sim here
            }

            results.append(result_row)
            pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)
