# H-MRS: Hybrid Moment-Ratio Scoring

This repository contains simulation code for the H-MRS algorithm proposed in the paper:

Yao Zhao (2026)
A Novel Hybrid Approach for Positive-Valed DAG Learning
CLeaR 2026

Files

hmrs.py – core implementation of the H-MRS algorithm

simulate_data.py – synthetic data generation and evaluation metrics

run_demo.py – example script demonstrating how to run H-MRS

requirements.txt – required Python packages

Requirements

Python 3.8+

Install dependencies with:

pip install -r requirements.txt

Example

Run the demo script:

python run_demo.py

This will:

1. Generate synthetic log-linear data
2. Run the H-MRS algorithm
3. Print estimated ordering, parents, and performance metrics
