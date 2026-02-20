# maze_qlearning.py
# Author: BOUTEROUATA Issam Salah Eddine
# Description: A Q-Learning agent that learns to navigate a simple grid maze.
# This is a classic Reinforcement Learning project. The agent starts at a
# fixed position, learns through trial and error which moves to make,
# and over time finds the shortest path to the goal.
#
# Concepts demonstrated:
#   - Reinforcement Learning (RL) fundamentals
#   - Q-Table (state-action value table)
#   - Epsilon-greedy exploration strategy
#   - Reward shaping

import random
import time

# ─────────────────────────────────────────
# MAZE DEFINITION
# 0 = open path, 1 = wall, S = start, G = goal
# Stored as a 2D grid (rows x cols)
# ─────────────────────────────────────────
MAZE = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
]

START = (0, 0)   # top-left
GOAL  = (4, 4)   # bottom-right

ROWS = len(MAZE)
COLS = len(MAZE[0])

# Actions: 0=Up, 1=Down, 2=Left, 3=Right
ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

# ─────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────
ALPHA        = 0.1     # learning rate — how much new info overrides old
GAMMA        = 0.9     # discount factor — how much future rewards matter
EPSILON      = 1.0     # starting exploration rate (100% random at first)
EPSILON_DECAY = 0.995  # how fast the agent stops exploring and starts exploiting
EPSILON_MIN  = 0.05    # minimum exploration rate
EPISODES     = 500     # number of training episodes


def init_q_table():
    """Create a Q-table with all zeros: states = grid cells, actions = 4 directions."""
    return {(r, c): [0.0, 0.0, 0.0, 0.0] for r in range(ROWS) for c in range(COLS)}


def get_next_state(state, action):
    """
    Given a state (row, col) and an action, return the next state.
    If the move hits a wall or goes out of bounds, stay in place.
    """
    r, c = state
    if action == 0:   # Up
        nr, nc = r - 1, c
    elif action == 1: # Down
        nr, nc = r + 1, c
    elif action == 2: # Left
        nr, nc = r, c - 1
    else:             # Right
        nr, nc = r, c + 1

    # Out of bounds or wall → stay
    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS or MAZE[nr][nc] == 1:
        return state
    return (nr, nc)


def get_reward(state):
    """Define the reward signal the agent receives for each state."""
    if state == GOAL:
        return 100    # big reward for reaching the goal
    else:
        return -1     # small penalty for each step (encourages efficiency)


def choose_action(q_table, state, epsilon):
    """
    Epsilon-greedy: with probability epsilon, explore randomly.
    Otherwise, exploit — choose the action with the highest Q-value.
    """
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return q_table[state].index(max(q_table[state]))


def train(q_table):
    """Run the Q-Learning training loop."""
    epsilon = EPSILON
    rewards_per_episode = []

    for episode in range(EPISODES):
        state = START
        total_reward = 0
        steps = 0
        max_steps = 200  # prevent infinite loops in early training

        while state != GOAL and steps < max_steps:
            action = choose_action(q_table, state, epsilon)
            next_state = get_next_state(state, action)
            reward = get_reward(next_state)

            # Q-Learning update rule
            best_next_q = max(q_table[next_state])
            old_q = q_table[state][action]
            q_table[state][action] = old_q + ALPHA * (reward + GAMMA * best_next_q - old_q)

            state = next_state
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if (episode + 1) % 100 == 0:
            avg = sum(rewards_per_episode[-100:]) / 100
            print(f"Episode {episode + 1:4d} | Avg reward (last 100): {avg:7.1f} | Epsilon: {epsilon:.3f}")

    return rewards_per_episode


def display_maze(path=None):
    """Print the maze to the terminal, optionally highlighting a path."""
    path_set = set(path) if path else set()
    print()
    for r in range(ROWS):
        row_str = ""
        for c in range(COLS):
            if (r, c) == START:
                row_str += " S "
            elif (r, c) == GOAL:
                row_str += " G "
            elif (r, c) in path_set:
                row_str += " · "
            elif MAZE[r][c] == 1:
                row_str += "███"
            else:
                row_str += " . "
        print(row_str)
    print()


def extract_path(q_table):
    """Follow the greedy policy from START to GOAL to extract the learned path."""
    state = START
    path = [state]
    visited = set()
    max_steps = 50

    while state != GOAL and len(path) < max_steps:
        if state in visited:
            return None  # loop detected — agent didn't learn a clean path
        visited.add(state)
        action = q_table[state].index(max(q_table[state]))
        state = get_next_state(state, action)
        path.append(state)

    return path if state == GOAL else None


def main():
    print("=" * 55)
    print("Maze Q-Learning Agent")
    print("Author: BOUTEROUATA Issam Salah Eddine")
    print("=" * 55)

    print("\nMaze layout (S=Start, G=Goal, ███=Wall):")
    display_maze()

    print(f"Training for {EPISODES} episodes...\n")
    q_table = init_q_table()
    train(q_table)

    print("\n" + "=" * 55)
    print("Training complete. Extracting learned path...")
    path = extract_path(q_table)

    if path:
        print(f"✅ Agent found a path in {len(path) - 1} steps!")
        print("\nLearned path through the maze (· = agent's route):")
        display_maze(path)
        print("Steps taken:")
        for i in range(1, len(path)):
            prev, curr = path[i - 1], path[i]
            dr = curr[0] - prev[0]
            dc = curr[1] - prev[1]
            if dr == -1: act = "Up"
            elif dr == 1: act = "Down"
            elif dc == -1: act = "Left"
            else: act = "Right"
            print(f"  {prev} → {curr}  ({act})")
    else:
        print("❌ Agent did not find a clean path. Try increasing EPISODES.")

    print("\nDone.")


if __name__ == "__main__":
    main()
