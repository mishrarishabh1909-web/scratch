from dataclasses import dataclass

@dataclass
class NodeSpec:
    node_id: int
    cpu_cap: float
    mem_cap: float
    bw_cap: float
    current_load: float = 0.0
    degraded: bool = False

class NodeManager:
    def __init__(self, config=None):
        self.config = config or {}
        num_nodes = self.config.get('num_fog_nodes', 5)
        self.nodes = {i: NodeSpec(i, 100.0, 100.0, 100.0) for i in range(1, num_nodes + 1)}

    def health_check(self):
        # Mock health check logic
        for n in self.nodes.values():
            n.degraded = False

    def rebalance(self):
        # Mock rebalance logic across nodes
        pass

    def add_node(self, node_id, cap):
        self.nodes[node_id] = NodeSpec(node_id, cap, cap, cap)

    def remove_node(self, node_id):
        if node_id in self.nodes:
            del self.nodes[node_id]
