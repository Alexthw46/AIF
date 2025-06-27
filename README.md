# 🍎 A*pple quest 🍎

## 🎯 Project Overview

This project explores classical Artificial Intelligence (GOFAI) search and planning algorithms within the [MiniHack](https://minihack.readthedocs.io/en/latest/) environment, a reinforcement learning platform built on top of NetHack. 
Our objective is to design a custom dungeon-like environment containing **apple** (reward), and evaluate different **search algorithms** to complete a task: **collect all apple while reaching the downstairs**.

## 🧪 Task Description

The environment is a procedurally generated maze containing:

    🟡 Apple tiles: provide positive rewards.

    🔽 Downstairs tile: marks the final goal.

    🟥 Step penalty: encourages efficiency.

The agent must compute an optimal path that maximizes rewards by collecting all apples and reaching the goal with minimum steps.
We test the algorithms both in an offline setting with full observability and in an online setting with partial observability given by walls inserted in the map.  

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

## 📊 Benchmarking & Evaluation

We compare each algorithm using the following metrics:

    ✅ Success rate (task completion)

    ⏱️ Planning time

    🧭 Path length

    🍎 Apples collected

    🏆 Total reward
    
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
