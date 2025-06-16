import random
from typing import Tuple, Set

import numpy as np

from utils import get_valid_moves, manhattan_distance


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # (position, collected_apples)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0


def is_terminal(state, target, apples):
    pos, collected = state
    return pos == target


def rollout_policy(game_map, state, target, apples):
    # Random rollout policy, simulate until terminal or max steps
    pos, collected = state
    collected = set(collected)
    steps = 0
    max_steps = 500  # limit rollout length
    while steps < max_steps:
        if pos == target:
            break
        moves = get_valid_moves(game_map, pos)
        if not moves:
            break
        pos = random.choice(moves)
        if pos in apples:
            collected.add(pos)
        steps += 1
    # Reward for collected apples + bonus if target reached after collecting all
    reward = len(collected)*0.75
    #if pos == target and collected == apples:
    #   reward += 10  # bonus reward for full succes
    if pos == target:
        reward += 1
    return reward  # reward = number of apples collected


def heuristic_rollout_policy(game_map, state, target, apples):
    pos, collected = state
    collected = set(collected)
    steps = 0
    max_steps = 100
    while steps < max_steps:
        if pos == target and collected == apples:
            break
        moves = get_valid_moves(game_map, pos)
        if not moves:
            break
        # Prioritize moves closer to the target or apples
        moves.sort(key=lambda move: (move in apples, -manhattan_distance(move, target)), reverse=True)
        pos = moves[0]
        if pos in apples:
            collected.add(pos)
        steps += 1
    reward = len(collected)
    if pos == target and collected == apples:
        reward += 10
    if pos == target:
        reward += 5
    return reward


def tree_policy(node, game_map, target, apples):
    # Select child with highest UCB1 or expand new child
    C = 1.4  # exploration constant
    if not node.children:
        return node
    best_score = -float('inf')
    best_child = None
    for child in node.children:
        if child.visits == 0 or node.visits == 0:
            score = float('inf')
        else:
            exploit = child.reward / child.visits
            explore = C * np.sqrt(np.log(node.visits) / child.visits)
            score = exploit + explore
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def expand(node, game_map, apples):
    pos, collected = node.state
    collected = set(collected)
    children_states = []
    for move in get_valid_moves(game_map, pos):
        new_collected = set(collected)
        if move in apples:
            new_collected.add(move)
        child_state = (move, frozenset(new_collected))
        if child_state not in [c.state for c in node.children]:
            children_states.append(child_state)
    for state in children_states:
        node.children.append(MCTSNode(state, parent=node))


def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent


def best_path(node, target: Tuple[int, int]) -> List[Tuple[int, int]]:
"""
def best_path(node):
    path = []
    while node:
        path.append(node.state[0])  # position only
        if node.children:
            # Choose child with the highest visits
            node = max(node.children, key=lambda c: c.visits)
        else:
            node = None
    return path
"""
    path = [node.state[0]]

    while node.children:
        # Filter children that are on a valid path to the target
        candidates = [child for child in node.children if reaches_target(child, target)]
        if not candidates:
            break
        node = max(candidates, key=lambda c: c.visits)
        path.append(node.state[0])

        if node.state[0] == target:
            break

    return path

def reaches_target(node: MCTSNode, target: Tuple[int, int]) -> bool:
    """Recursively checks if any descendant of the node ends at the target."""
    if node.state[0] == target:
        return True
    return any(reaches_target(child, target) for child in node.children)


def mcts(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], apples: Set[Tuple[int, int]], iterations=1000, policy=rollout_policy):
    root = MCTSNode((start, frozenset()))
    
    for _ in range(iterations):
        node = root
        # Selection
        while node.children:
            node = tree_policy(node, game_map, target, apples)
        # Expansion
        if not is_terminal(node.state, target, apples):
            expand(node, game_map, apples)
            if node.children:
                node = random.choice(node.children)
        # Simulation
        reward = policy(game_map, node.state, target, apples)
        # Backpropagation
        backpropagate(node, reward)

    return best_path(root, target)
