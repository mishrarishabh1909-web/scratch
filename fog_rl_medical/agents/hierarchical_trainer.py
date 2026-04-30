import numpy as np

class HierarchicalTrainer:
    def __init__(self, high_policy, low_policies, env):
        self.high_policy = high_policy
        self.low_policies = low_policies  # dict mapping node_id -> LowLevelPolicy
        self.env = env
        
    def train_step(self, priority_tasks):
        state = self.env.state
        
        # 1. High-level assigns tasks
        assignments = self.high_policy.select_node(state, priority_tasks)
        
        # 2. Low-level allocates resources for assigned tasks
        allocations = {}
        for task in priority_tasks:
            node_id = assignments.get(task.task_id)
            if node_id in self.low_policies and node_id > 0: # 0 is cloud
                lp = self.low_policies[node_id]
                node_slice = np.array([
                    state.cpu_utilization[node_id-1],
                    state.memory_utilization[node_id-1],
                    state.bandwidth_utilization[node_id-1],
                    state.queue_depths[node_id-1],
                    state.sla_violations[node_id-1]
                ])
                alloc = lp.allocate(node_slice, task.shared_embedding)
                allocations[task.task_id] = alloc
                
        # 3. Environment Step
        next_state, reward, done, info = self.env.step(assignments, allocations, priority_tasks)
        return next_state, reward, done
