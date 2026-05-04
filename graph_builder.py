# graph_builder.py
import networkx as nx

class InteractionGraphBuilder:
    def __init__(self, alpha=0.5, beta=0.5):
        """
        alpha, beta: weighting parameters for edge weight formula
        wij = alpha * fij + beta * lij [1]
        """
        self.alpha = alpha
        self.beta = beta

    def build_graph(self, interaction_logs, telemetry):
        """
        Constructs directed weighted graph G=(V, E) [1].
        Nodes = services, Edges = interactions with weight wij.
        """
        G = nx.DiGraph()

        # Add nodes with telemetry attributes
        for _, row in telemetry.iterrows():
            G.add_node(row["service"],
                       mean_latency=row["mean_latency"],
                       error_rate=row["error_rate"],
                       cpu_utilization=row["cpu_utilization"],
                       memory_usage=row["memory_usage"])

        # Add weighted edges using wij = alpha * fij + beta * lij [1]
        for _, row in interaction_logs.iterrows():
            weight = (self.alpha * row["frequency"] +
                      self.beta * row["latency"])
            G.add_edge(row["source"], row["target"], weight=weight,
                       frequency=row["frequency"],
                       latency=row["latency"])

        return G