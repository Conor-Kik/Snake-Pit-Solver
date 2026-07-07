# Snake Pit Puzzle – Integer Programming Formulations

This repository contains Python implementations of three integer programming formulations for solving the **Snake Pit Puzzle**, implemented using **Gurobi 13.0.2**.

The Snake Pit puzzle is a region-division logic puzzle in which a grid must be partitioned into disjoint one-cell-wide paths ("snakes") subject to length, endpoint, and equal-length adjacency constraints.

The code is intended as supplementary teaching material for a paper on using the Snake Pit puzzle to teach formulation choice and comparative modelling in integer programming.

Puzzle source and rules:  
https://www.gmpuzzles.com/blog/category/regiondivision/snake-pit/

## Formulations

The repository includes three formulations:

- `snakepitLazy.py`  
  Square-assignment formulation with lazy constraints.

- `snakepitFlow.py`  
  Flow-based formulation using directed arcs and ordering variables.

- `snakepitCG.py`  
  Path-generation/set-partitioning formulation using pre-generated snake paths and lazy constraints for equal-length adjacency.

Supporting files:

- `problems.py`  
  Defines the puzzle instances.

- `problems/`  
  Contains PDF versions of the Snake Pit puzzle instances for reference and classroom use.

- `board_plotting.py`  
  Contains plotting utilities for displaying puzzle grids and solutions.

## Requirements

- Python 3.x
- Gurobi 13.0.2
- A valid Gurobi license
- Matplotlib

## How to Run

Each formulation is designed to be run directly from the file.

For example:

```bash
python snakepitLazy.py
python snakepitFlow.py
python snakepitCG.py