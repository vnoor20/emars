# generate_figures.py
import matplotlib.pyplot as plt
import numpy as np
from baselines import (PerServiceSRE, RandomGraphPartitioning,
                       ADAFBaseline, EMARSSystem)

workloads = [100, 200, 300, 400, 500, 600, 700, 800]

b1 = PerServiceSRE()
b2 = RandomGraphPartitioning()
b3 = ADAFBaseline()
b4 = EMARSSystem()

latencies = {
    "Per-Service SRE": [],
    "Random Partitioning (RGP)": [],
    "ADAF": [],
    "E-MARS": []
}

for w in workloads:
    latencies["Per-Service SRE"].append(b1.evaluate(w)[0])
    latencies["Random Partitioning (RGP)"].append(b2.evaluate(w)[0])
    latencies["ADAF"].append(b3.evaluate(w)[0])
    latencies["E-MARS"].append(b4.evaluate(w)[0])

# ── Figure 2: Latency vs Workload ──────────────────────────────────────
plt.figure(figsize=(10, 6))
styles = {
    "Per-Service SRE":        ("r",      "o"),
    "Random Partitioning (RGP)": ("orange", "s"),
    "ADAF":                   ("b",      "^"),
    "E-MARS":                 ("g",      "D")
}
for label, values in latencies.items():
    color, marker = styles[label]
    plt.plot(workloads, values, color=color, marker=marker,
             linestyle="-", linewidth=2, label=label)

plt.xlabel("Workload (Requests per Second)", fontsize=12)
plt.ylabel("Average Latency (ms)", fontsize=12)
plt.title("Figure 2: Latency vs Workload — All Configurations", fontsize=14)
plt.legend(loc="upper left", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(workloads)
plt.tight_layout()
plt.savefig("figure2_latency_comparison.png", dpi=300)
plt.show()
print("[Saved] figure2_latency_comparison.png")

# ── Figure 3: Multi-Metric Bar Chart ───────────────────────────────────
metrics = ["MTTR\n(min)", "Error Rate\n(%)",
           "Resource\nUtil (%)", "Latency\nReduction (%)",
           "Failure\nPropagation\n(1=Low,4=High)"]

values = {
    "Per-Service SRE":           [18, 2.3, 63, 0,  4],
    "Random Partitioning (RGP)": [15, 2.0, 68, 10, 3],
    "ADAF":                      [13, 1.8, 72, 18, 2],
    "E-MARS":                    [11, 1.4, 78, 28, 1]
}

x = np.arange(len(metrics))
width = 0.2
colors = ["red", "orange", "blue", "green"]

fig, ax = plt.subplots(figsize=(13, 7))
for i, (label, vals) in enumerate(values.items()):
    ax.bar(x + (i - 1.5) * width, vals, width,
           label=label, color=colors[i], alpha=0.85)

ax.set_xlabel("Performance Metric", fontsize=12)
ax.set_ylabel("Value", fontsize=12)
ax.set_title("Figure 3: Multi-Metric Comparison — All Configurations",
             fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figure3_multibaseline_comparison.png", dpi=300)
plt.show()
print("[Saved] figure3_multibaseline_comparison.png")