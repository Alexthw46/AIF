# 🍎 A*pple quest 🍎

## 🎯 Project Overview

This project explores classical Artificial Intelligence (GOFAI) search and planning algorithms within the [MiniHack](https://minihack.readthedocs.io/en/latest/) environment, a reinforcement learning platform built on top of NetHack. 
Our objective is to design a custom dungeon-like environment containing **apple** (reward), and evaluate different **search algorithms** to complete a task: **collect all apple while reaching the downstairs**.

## 🧪 Task Description

The agent starts in a procedurally generated maze populated with:
- 🟡 **apple tiles** to collect (positive reward)

The challenge is to **plan an optimal path** to collect all apple using only classical search algorithms—no learning involved.

## 🛠️ What We Do

- 🔧 **Custom Environment Design**: We define a personalized MiniHack map with structured room layouts, apple locations, reward manager.
- 🔍 **Algorithm Implementation**: We implement and test multiple search-based planning algorithms:
  - **A\* Search**
  - **Online A\***
  - **Weighted A\***
  - **Best-first search (greedy variant) A\***
  - **Monte Carlo Tree Search (MCTS) A\***
  - **Potential Fields A\***
  - **Beam search A\***
  - 
- 📊 **Benchmarking**: Algorithms are compared across various metrics:
  - Success rate
  - Time to plan
  - Path length
  - apple collected
  - reward 
## 📁 Project Structure

```bash
.
├── algorithms/          # Search algorithms 
├── algorithms_online/    # Online search algorithms
├── MCTS.py/              # Monte Carlo Tree search implementation 
├── Benchmark_Offline/    # Evaluation results and logs (Offline setting)
├── Benchmark_Online/     # Evaluation results and logs (Online setting) 
├── report/              # Final project report
├── simulator.py/         # Final project report
├── report/              # Final project report
└── README.md            # This file
