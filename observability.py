# observability.py
import numpy as np
import pandas as pd

class ObservabilityLayer:
    def __init__(self, num_services=50):
        self.num_services = num_services
        self.services = [f"s{i}" for i in range(num_services)]

    def collect_telemetry(self):
        """
        Simulates runtime telemetry collection per service.
        Mirrors Prometheus / OpenTelemetry data collection [1].
        """
        data = []
        for service in self.services:
            data.append({
                "service": service,
                "mean_latency": np.random.uniform(50, 300),       # ms
                "error_rate": np.random.uniform(0.01, 0.05),      # %
                "cpu_utilization": np.random.uniform(0.3, 0.9),   # ratio
                "memory_usage": np.random.uniform(0.2, 0.85),     # ratio
                "throughput": np.random.uniform(100, 800)          # req/s
            })
        return pd.DataFrame(data)

    def collect_interaction_logs(self):
        """
        Simulates service-to-service interaction logs.
        Each record captures frequency and latency between two services [1].
        """
        logs = []
        for i in range(self.num_services):
            num_deps = np.random.randint(3, 6)
            targets = np.random.choice(
                [j for j in range(self.num_services) if j != i],
                size=num_deps,
                replace=False
            )
            for t in targets:
                logs.append({
                    "source": f"s{i}",
                    "target": f"s{t}",
                    "frequency": np.random.uniform(10, 500),
                    "latency": np.random.uniform(5, 150)
                })
        return pd.DataFrame(logs)