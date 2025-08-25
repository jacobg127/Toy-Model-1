# Toy-Model-1: Emergent Arrow of Time from Coarse-Graining

This repository contains a simple Python simulation that illustrates how an **arrow of time** can emerge from fully reversible microscopic dynamics when observed through coarse-graining.

## Description
- The model evolves a 2D binary lattice under a fixed **permutation** (bijective and reversible).
- At the microscopic level, all information is preserved — no entropy increase.
- When the system is observed only in **coarse-grained blocks**, the measured entropy rises rapidly and stabilizes, mimicking the macroscopic arrow of time.
- Reversing the permutation exactly retraces the dynamics, proving that irreversibility is an artifact of coarse-graining.

This serves as a toy model for the hypothesis that **time and entropy are emergent phenomena** arising from observer coarse-graining, inspired by ideas from loop quantum gravity and relational quantum mechanics.

## Requirements
- Python 3.8+
- `numpy`
- `matplotlib`

You can install dependencies with:
```bash
pip install numpy matplotlib
