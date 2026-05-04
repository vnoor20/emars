# main.py
from observability import ObservabilityLayer
from graph_builder import InteractionGraphBuilder
from module_detection import ModuleDetectionEngine
from reliability_control import ReliabilityControlEngine
from rl_engine import RLEnvironment, DQNAgent, PPOAgent
import generate_figures  # runs figure generation on import

def run_emars(num_services=50):
    print("=" * 60)
    print("  E-MARS: Emergent Modular Adaptive Reliability System")
    print("=" * 60)

    # Stage 1 — Observability [1]
    print("\n[Stage 1] Collecting telemetry...")
    obs = ObservabilityLayer(num_services=num_services)
    telemetry = obs.collect_telemetry()
    logs = obs.collect_interaction_logs()
    print(f"  Services: {num_services} | Interactions: {len(logs)}")

    # Stage 2 — Graph Construction [1]
    print("\n[Stage 2] Building interaction graph...")
    builder = InteractionGraphBuilder(alpha=0.5, beta=0.5)
    G = builder.build_graph(logs, telemetry)
    print(f"  Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    # Stage 3 — Module Detection [1]
    print("\n[Stage 3] Detecting emergent modules...")
    detector = ModuleDetectionEngine(threshold=0.001)
    modules, q_score = detector.detect_modules(G)
    print(f"  Modules Discovered: {len(modules)}")
    print(f"  Modularity Score Q: {q_score:.4f}")

    # Stage 4 — Reliability Control [1]
    print("\n[Stage 4] Evaluating module health...")
    control = ReliabilityControlEngine(health_threshold=0.6)
    health_scores = control.evaluate_all_modules(modules, telemetry)
    actions = control.apply_reliability_policies(health_scores)

    unhealthy = [a for a in actions if a["action"] != "nominal"]
    print(f"  Modules requiring recovery: {len(unhealthy)}")

    # Stage 5 — RL Engine [1]
    print("\n[Stage 5] Initializing RL agents...")
    env = RLEnvironment(num_modules=len(modules))
    dqn = DQNAgent(env.state_dim, env.action_dim)
    ppo = PPOAgent(env.state_dim, env.action_dim)

    state = env.get_state(health_scores, telemetry)
    dqn_action = dqn.select_action(state)
    ppo_action = ppo.select_action(state)

    action_map = {0: "Scale Up", 1: "Scale Down", 2: "Isolate"}
    print(f"  DQN (reactive) action:  {action_map[dqn_action]}")
    print(f"  PPO (strategic) action: {action_map[ppo_action]}")

    print("\n[E-MARS] Pipeline complete.")
    print("[Figures] Generating Figure 2 and Figure 3...")

if __name__ == "__main__":
    run_emars(num_services=50)