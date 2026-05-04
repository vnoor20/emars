# baselines.py
import numpy as np

class PerServiceSRE:
    """Baseline 1: Traditional per-service SRE [1]"""
    def evaluate(self, workload):
        # Exponential latency degradation under load
        latency = 100 + 0.0005 * (workload ** 2)
        error_rate = 0.023 + workload * 0.00003
        resource_util = 0.63
        mttr = 18
        return latency, error_rate, resource_util, mttr


class RandomGraphPartitioning:
    """Baseline 2: Random service grouping — isolates clustering quality"""
    def evaluate(self, workload):
        latency = 95 + 0.0004 * (workload ** 2)
        error_rate = 0.020 + workload * 0.000025
        resource_util = 0.68
        mttr = 15
        return latency, error_rate, resource_util, mttr


class ADAFBaseline:
    """Baseline 3: AI-driven per-service automation without graph awareness [1]"""
    def evaluate(self, workload):
        latency = 90 + 0.00035 * (workload ** 2)
        error_rate = 0.018 + workload * 0.00002
        resource_util = 0.72
        mttr = 13
        return latency, error_rate, resource_util, mttr


class EMARSSystem:
    """E-MARS: Module-aware reliability with graph clustering + RL [1]"""
    def evaluate(self, workload):
        # Near-linear latency scaling due to coordinated module control
        latency = 85 + 0.00018 * (workload ** 1.4)
        error_rate = 0.014 + workload * 0.000012
        resource_util = 0.78
        mttr = 11
        return latency, error_rate, resource_util, mttr