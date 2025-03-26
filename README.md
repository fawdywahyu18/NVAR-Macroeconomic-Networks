# NVAR-Macroeconomic-Networks

This repository contains Python code for implementing the **Network Vector Autoregression (NVAR)** model as proposed in the paper:  
**"Cross-Sectional Dynamics Under Network Structure: Theory and Macroeconomic Applications"** by Marko Mlikota (Geneva Graduate Institute).

## 📖 Description
The NVAR framework provides a flexible approach for modeling the dynamics of cross-sectional variables influenced by network structures. The model allows for innovation transmission along bilateral links and accommodates high-order network effects that accumulate over time. This repository includes tools for:
- Estimating dynamic network effects from given networks.
- Inferring network structures from dynamic cross-correlations in data.
- Dimensionality-reduction techniques for high-dimensional processes.
- Implementing Granger-causality tests under the NVAR framework.
- Forecasting macroeconomic aggregates using network structures.

## 🚀 Features
- Implementation of **NVAR(p,1)** and **NVAR(p,q)** models.
- Estimation methods using **Ordinary Least Squares (OLS)** and **Generalized Least Squares (GLS)**.
- Joint inference for estimating network structures and effect-timing.
- Tools for simulating impulse-responses and forecasting.

## 📂 Structure
- `src/`: Python code implementing the NVAR models.
- `examples/`: Notebooks demonstrating model usage with sample data.
- `docs/`: Detailed documentation and theoretical background.
- `tests/`: Unit tests to ensure robustness of the implementation.

## 📜 Citation
If you use this code, please cite the original paper:  
Mlikota, M. (2025). *Cross-Sectional Dynamics Under Network Structure: Theory and Macroeconomic Applications*. Geneva Graduate Institute.
