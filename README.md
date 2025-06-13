# 🧠 ???? A TITLE for the project ????

## 🎯 Project Overview

This project explores classical Artificial Intelligence (GOFAI) search (and planning algorithms??) within the [MiniHack](https://minihack.readthedocs.io/en/latest/) environment, a reinforcement learning platform built on top of NetHack. 
Our objective is to design a custom dungeon-like environment containing **apple** (reward) and **monsters** (threats), and evaluate different **search algorithms** to complete a task: **collect all apple while avoiding monsters**.

## 🧪 Task Description

The agent starts in a procedurally generated maze populated with:
- 🟡 **apple tiles** to collect (positive reward)
- 👾 **Monsters** to avoid (negative consequence or terminal state)

The challenge is to **plan a safe and optimal path** to collect all apple using only classical search algorithms—no learning involved.

## 🛠️ What We Do

- 🔧 **Custom Environment Design**: We define a personalized MiniHack map with structured room layouts, apple locations, and monster hazards.
- 🔍 **Algorithm Implementation**: We implement and test multiple search-based planning algorithms:
  - **Breadth-First Search (BFS)**
  - **A\* Search**
  - **Online A\*** (recompute path every step)
  - **Weighted A\***
  - *(Optionally)* Lifelong Planning A\*, Real-Time A\*, or ARA\*( good for time-bounded planning, Best-first search (greedy variant), Monte Carlo Tree Search (MCTS) as contrast to classical or Potential Fields (use heuristics like inverse distance from monster)
- 📊 **Benchmarking**: Algorithms are compared across various metrics:
  - Success rate
  - Time to plan
  - Path length
  - apple collected

## 📁 Project Structure

```bash
.
├── env/                 # Custom MiniHack environment code
├── agents/              # Search algorithms
├── runner.py            # Main experiment script
├── benchmarks/          # Evaluation results and logs
├── report/              # Final project report
└── README.md            # This file
