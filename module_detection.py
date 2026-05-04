# module_detection.py
import networkx as nx
import networkx.algorithms.community as nx_comm

class ModuleDetectionEngine:
    def __init__(self, threshold=0.001):
        self.threshold = threshold
        self.modules = []

    def detect_modules(self, G):
        """
        Applies Louvain-style greedy modularity maximization [1].
        Maximizes modularity score Q to find emergent service modules.
        Complexity: O(N log N) [1]
        """
        # Convert to undirected for community detection
        G_undirected = G.to_undirected()

        # Greedy modularity maximization — Louvain-style [1]
        communities = nx_comm.greedy_modularity_communities(
            G_undirected, weight="weight"
        )

        self.modules = [list(c) for c in communities]

        # Compute modularity score Q
        modularity_score = nx_comm.modularity(
            G_undirected,
            communities,
            weight="weight"
        )

        print(f"[Module Detection] Found {len(self.modules)} modules")
        print(f"[Module Detection] Modularity Score Q = {modularity_score:.4f}")

        return self.modules, modularity_score

    def get_module_assignments(self):
        """Returns a dict mapping service -> module_id"""
        assignments = {}
        for module_id, module in enumerate(self.modules):
            for service in module:
                assignments[service] = module_id
        return assignments