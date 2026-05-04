# reliability_control.py
import numpy as np

class ReliabilityControlEngine:
    def __init__(self, health_threshold=0.6):
        """
        health_threshold: minimum acceptable module health score [1]
        """
        self.health_threshold = health_threshold

    def compute_module_health(self, module, telemetry_df):
        """
        Computes module health function H(m) [1].
        H(m) = mean reliability score across all services in module.
        Reliability score = 1 - error_rate (normalized).
        """
        module_data = telemetry_df[telemetry_df["service"].isin(module)]
        if module_data.empty:
            return 1.0
        reliability_scores = 1 - module_data["error_rate"].values
        return float(np.mean(reliability_scores))

    def evaluate_all_modules(self, modules, telemetry_df):
        """Evaluates health of all detected modules [1]"""
        health_scores = {}
        for idx, module in enumerate(modules):
            health_scores[f"module_{idx}"] = {
                "services": module,
                "health": self.compute_module_health(module, telemetry_df)
            }
        return health_scores

    def apply_reliability_policies(self, health_scores):
        """
        Triggers coordinated recovery when module health
        falls below threshold [1].
        Actions: scale_up, isolate, reroute_traffic
        """
        actions = []
        for module_id, info in health_scores.items():
            health = info["health"]
            services = info["services"]

            if health < self.health_threshold:
                print(f"[Control] {module_id} health={health:.3f} "
                      f"— BELOW THRESHOLD. Triggering recovery.")
                actions.append({
                    "module": module_id,
                    "services": services,
                    "action": "scale_up_and_isolate",
                    "health": health
                })
            else:
                actions.append({
                    "module": module_id,
                    "services": services,
                    "action": "nominal",
                    "health": health
                })

        return actions