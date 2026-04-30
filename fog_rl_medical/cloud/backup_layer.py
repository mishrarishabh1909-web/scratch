class BackupLayer:
    def __init__(self):
        self.offloaded_tasks = 0
        self.total_tasks = 0

    def should_offload(self, task, cluster_state):
        # Trigger condition: any fog node CPU > 90%
        # Offload P3/P4 tasks to cloud
        self.total_tasks += 1
        overloaded = any(cpu > 0.9 for cpu in cluster_state.cpu_utilization)
        if overloaded and task.priority in [3, 4]:
            self.offloaded_tasks += 1
            return True
        return False

    def get_offload_ratio(self):
        if self.total_tasks == 0:
            return 0.0
        return self.offloaded_tasks / self.total_tasks
