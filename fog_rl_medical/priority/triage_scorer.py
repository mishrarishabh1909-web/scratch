import numpy as np

class TriageScorer:
    def __init__(self, config=None):
        self.config = config or {}
        # weights: rule_score: 0.6, ml_score: 0.4
        self.w_rule = 0.6
        self.w_ml = 0.4

    def score(self, task_embedding):
        # Mock rule-based priority with more balanced distribution
        rule_priority = np.random.choice([1, 2, 3, 4], p=[0.25, 0.25, 0.25, 0.25])
        
        # Mock ML-based urgency scorer with different distribution
        ml_priority = np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2])
        
        # Combine weighted average
        combined = int(np.round(self.w_rule * rule_priority + self.w_ml * ml_priority))
        combined = max(1, min(4, combined))
        
        confidence = np.random.uniform(0.7, 0.95)  # Varied confidence
        return combined, confidence
