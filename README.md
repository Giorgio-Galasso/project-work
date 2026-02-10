# Vehicle Routing Problem (VRP) with Load-Dependent Costs

## Overview

This repository contains a Python implementation of a **Hybrid Memetic Algorithm** designed to solve a variation of the Vehicle Routing Problem (VRP) where the travel cost is non-linear and dependent on the carried load.

The solution is highly optimized using **Numba** for JIT compilation and features an adaptive strategy that changes behavior based on the problem's structural parameters (alpha and beta).

## Key Features

* **Adaptive Strategy**
    * **Low Penalty (beta <= 1):** Uses a Genetic Algorithm with **Large Neighborhood Search (LNS)** and **2-opt** local search to optimize geometry.
    * **High Penalty (beta > 1):** Uses a **Bulk Removal** preprocessing step to handle heavy loads via dedicated trips, focusing on weight minimization.

* **Prins Split Algorithm**
    Efficiently converts a TSP-like permutation (Giant Tour) into valid VRP trips using a shortest-path approach on a DAG.

* **Hybrid Operators**
    Includes IOX Crossover, Inversion/Swap Mutation, and Smart LNS (Destroy & Repair).

* **High Performance**
    Critical path evaluations are accelerated using `numba` and `numpy`.

## Project Structure

```text
.
├── Problem.py              # Problem generator class
├── s319497.py              # Main entry point (Solution function)
├── base_requirements.txt   # Dependencies
└── src/
    ├── algorithms.py       # Core logic (Split, GA operators, LNS) - JIT compiled
    ├── evaluator.py        # SmartEvaluator class and preprocessing logic
    └── utils.py            # Mathematical cost 
```

## Requirements
To run the code, install the required dependencies:
```text
pip install -r base_requirements.txt
```

## Usage
The solution is encapsulated in the ```solution(p: Problem)``` function within ```s319497.py```. It takes a ```Problem``` instance as input and returns the optimal path as a list of tuples in the format ```[(node, gold_collected), ...]```.

## Acknowledgments
The code documentation, docstrings, and comments throughout this project were generated and refined with the assistance of Google Gemini.