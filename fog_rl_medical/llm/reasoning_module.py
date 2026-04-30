import time
import random

class ReasoningModule:
    def __init__(self, config=None):
        self.config = config or {}
        self.provider = self.config.get('provider', 'openai')

    def query(self, prompt, timeout_ms=2000):
        # Mock LLM API call with a small sleep to simulate network
        time.sleep(0.01)
        
        # Generate varied priorities based on rule_priority from prompt
        # Extract rule_priority from prompt (this is a simple mock)
        import random
        rule_priority = 2  # Default fallback
        if "rule_priority" in prompt:
            # Simple extraction - in real implementation this would parse properly
            pass
        
        # Return varied priorities with different confidence levels
        priorities = [1, 2, 3, 4]
        weights = [0.15, 0.35, 0.35, 0.15]  # Slightly favor mid-range priorities
        priority = random.choices(priorities, weights=weights)[0]
        
        # Sometimes return low confidence to use rule-based priority
        confidence = random.uniform(0.6, 0.95)
        
        rationales = [
            'Patient shows stable vitals, low priority needed.',
            'Moderate symptoms detected, standard priority assigned.',
            'Elevated risk factors, increased priority recommended.',
            'Critical condition indicators, highest priority required.'
        ]
        
        return {
            'priority': priority,
            'rationale': rationales[priority - 1],
            'confidence': confidence
        }
