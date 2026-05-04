# E-MARS: Emergent Modular Adaptive Reliability System

A data-driven reliability management framework for microservice
architectures using graph-based community detection and reinforcement
learning.

## Authors
Rahul Dhiman & Harnoor Kaur
Chitkara University, Punjab, India
Mentor: Dr. Anshu Singla

## Installation
pip install -r requirements.txt

## Usage
python main.py

## Project Structure
- observability.py     — Telemetry collection (Stage 1)
- graph_builder.py     — Interaction graph construction (Stage 2)
- module_detection.py  — Emergent module detection (Stage 3)
- reliability_control.py — Module health & control (Stage 4)
- rl_engine.py         — DQN + PPO agents (Stage 5)
- baselines.py         — All four baseline configurations
- generate_figures.py  — Generates Figure 2 and Figure 3
- main.py              — Main pipeline runner

## Results
| Metric | Baseline | E-MARS | Improvement |
|--------|----------|--------|-------------|
| MTTR | 18 min | 11 min | 39% |
| Error Rate | 2.3% | 1.4% | 39% |
| Resource Utilization | 63% | 78% | +15% |
| Mean Latency | — | — | -28% |
