import time

import IPython.display as display
import gymnasium as gym
import matplotlib.pyplot as plt
from minihack import LevelGenerator
from minihack import RewardManager
from nle import nethack

from MCTS import *
from utils import *
from utils import get_player_location


def stairs_reward_function(env, previous_observation, action, observation):
    # Agent is on stairs down
    if observation[env._internal_index][4]:
        return 1
    return 0


def simulate_with_heuristic(env, fun: callable, clear_outputs=True, wait_time: float = 0.5, cropped=True, **kwargs):
    """
    Simulates the static environment using a heuristic function.

    Parameters:
    - env: The environment to simulate.
    - fun: The heuristic function to use for simulation. Need to follow the signature:
        fun(game_map, start, target, apple_positions, **kwargs)
    - kwargs: Additional keyword arguments for the heuristic function.

    Returns:
    - The result of the simulation.
    """

    state, _ = env.reset()
    game_map = state['chars']
    game = state['pixel_crop' if cropped else 'pixel']
    start = get_player_location(game_map)
    target = get_target_location(game_map)

    char_map = np.vectorize(chr)(game_map)
    apple_positions = np.where(char_map == '%')

    # zip into a list of int tuples (x, y)
    apple_positions = list(zip(apple_positions[0], apple_positions[1]))
    print("Apple positions:", [(int(x), int(y)) for x, y in apple_positions])

    path = fun(game_map, start, target, set(apple_positions), **kwargs)

    if path is None or len(path) == 0:
        print("No path found.")
        return 0
    else:
        print("Path found:")
        for (x, y) in path:
            print(x, y)

    actions = actions_from_path(start, path[1:])
    simulate_path(path, game_map, actions)

    image = plt.imshow(game if cropped else game[:, 400:850])

    time.sleep(2)
    tot_reward = 0
    done = False
    for action in actions:
        s, reward, done, _, dic = env.step(action)
        tot_reward += reward
        if not done:
            # check and eat apple
            s, reward, _, _, _ = check_and_eat_apple(s, env, apple_positions)
            tot_reward += reward
            display.display(plt.gcf())
            print("Reward:", tot_reward)
            if clear_outputs:
                display.clear_output(wait=True)
            time.sleep(wait_time)
            image.set_data(s['pixel_crop'] if cropped else s['pixel'][:, 300:975])
            print("Action taken:", directions[action])
        else:
            print(f"Episode finished:", dic)
            print("Reward:", tot_reward)
            break
    print_path_on_map(game_map, path)
    if not done:
        print("Stairs were not reached.")
    return tot_reward


def simulate_online(env, fun: callable, clear_outputs=True, wait_time: float = 0.5, cropped=True, **kwargs):
    """
    Simulates the dynamic environment using a heuristic function.

    Parameters:
    - env: The environment to simulate.
    - fun: The heuristic function to use for simulation. Need to follow the signature:
        fun(game_map, start, **kwargs)
    - kwargs: Additional keyword arguments for the heuristic function.

    Returns:
    - The result of the simulation.
    """

    state, _ = env.reset()
    game_map = state['chars']
    game = state['pixel_crop' if cropped else 'pixel']

    image = plt.imshow(game if cropped else game[:, 400:850])

    time.sleep(2)
    tot_reward = 0
    done = False
    dic = {}
    s = state
    old_apple_positions = []
    while not done:
        start = get_player_location(game_map)

        print("Evaluating path...")
        # choose a target location and path to it
        path = fun(game_map, start, **kwargs)

        actions = actions_from_path(start, path[1:]) if path is not None else []
        simulate_path(path, game_map, actions)
        time.sleep(wait_time)
        for action in actions:
            # check where the apples are before moving
            apple_positions = np.where(np.vectorize(chr)(s['chars']) == '%')
            apple_positions = list(zip(apple_positions[0], apple_positions[1]))
            # a new apple has been discovered
            if len(apple_positions) > len(old_apple_positions):
                old_apple_positions = apple_positions
                print("Apple discovered, recomputing...")
                break
            s, reward, done, _, dic = env.step(action)
            tot_reward += reward
            display.display(plt.gcf())
            time.sleep(wait_time)
            if clear_outputs:
                display.clear_output(wait=True)
            print("Reward:", tot_reward)
            image.set_data(s['pixel_crop'] if cropped else s['pixel'][:, 300:975])
            print("Action taken:", directions[action])

            if not done:
                # check if we are on an apple and eat it
                s, reward, _, _, _ = check_and_eat_apple(s, env, apple_positions)
                tot_reward += reward
            else:
                break
            old_apple_positions = apple_positions
            game_map = s['chars']

    print(f"Episode finished:", dic)
    print("Reward:", tot_reward)

    return tot_reward


def check_and_eat_apple(state, env, apple_positions):
    """
    Check if the player is on an apple and eat it.
    """

    PICKUP = env.unwrapped.actions.index(nethack.Command.PICKUP)
    EAT = env.unwrapped.actions.index(nethack.Command.EAT)
    # INVENTORY = env.unwrapped.actions.index(nethack.Command.INVENTORY)

    game_map = state['chars']
    player_location = get_player_location(game_map)
    print(f"Player location: {player_location}")
    print(f"Apple location: {apple_positions}")
    if player_location in apple_positions:
        # Pick up the apple
        print("Found apple at:", player_location)
        s, _, done, _, dic = env.step(PICKUP)
        print("ACTION_TAKEN: PICKUP")
        apple_positions.remove(player_location)  # Remove the apple from the list
        s, _, done, _, dic = env.step(EAT)
        # What do you want to eat?[g or *]
        msg = bytes(s['message']).decode('utf-8').rstrip('\x00')
        print(msg)
        food_char = msg.split('[')[1][0]  # Because of the way the message in NetHack works

        # print("Food character to eat:", food_char)
        # Eat the apple
        # Open inventory to find the apple letter
        s, reward, done, boh, dic = env.step(env.unwrapped.actions.index(ord(food_char)))
        print("ACTION_TAKEN: EAT", food_char)
        msg = bytes(s['message']).decode('utf-8').rstrip('\x00')
        print(msg)
        return s, reward, done, boh, dic

    # If not on an apple, just return the state
    return state, 0, False, None, None


def create_env(map, penalty_time: float = -0.1, apple_reward: float = 0.75) -> gym.Env:
    """
    Create a MiniHack environment with the specified map and reward settings.
    """

    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    ACTIONS = MOVE_ACTIONS + (nethack.Command.EAT, nethack.Command.PICKUP, nethack.Command.INVENTORY) + tuple(
        range(ord('a'), ord('z') + 1))

    # Define a reward manager
    reward_manager = RewardManager()
    # Rward for eating an apple
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


def make_map(map_str: str, n_apples: int, seed=None) -> str:
    """
    Create a map file for the MiniHack environment.
    """
    lvl_gen = LevelGenerator(map=map_str)
    cols, rows = lvl_gen.x, lvl_gen.y

    # Fix the start position if a seed is provided
    if seed is not None:
        lvl_gen.set_start_pos((1, 1))

    # Generate random positions for apples
    if seed is None:
        for pos in range(0, n_apples):
            lvl_gen.add_object("apple", "%")
    else:
        apple_positions = randomize_apple_positions(map_str, 1, 2, cols - 1, rows - 2, n_apples, seed=seed)

        for pos in apple_positions:
            lvl_gen.add_object("apple", "%", place=pos)

    # Randomly place the stairs down if no seed is provided
    if seed is not None:
        lvl_gen.add_stair_down((cols - 2, rows - 2))
    else:
        lvl_gen.add_stair_down()

    return lvl_gen.get_des()
