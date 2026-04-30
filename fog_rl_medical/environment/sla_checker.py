from fog_rl_medical.fog.sla_profiles import SLA_PROFILES

class SLAChecker:
    def __init__(self):
        pass

    def evaluate(self, task):
        """
        Evaluates if a completed task met its SLA.
        Expected task dict keys: 'priority', 'total_latency', 'compute_allocated'
        Returns: bool indicating if SLA is met.
        """
        priority = task.get('priority', 4)
        profile = SLA_PROFILES.get(priority)
        
        if not profile:
            return False
            
        latency_met = task.get('total_latency', float('inf')) <= profile.latency_ms
        compute_met = task.get('compute_allocated', 0.0) >= profile.compute_budget
        
        return latency_met and compute_met
