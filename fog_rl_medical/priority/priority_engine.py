from dataclasses import dataclass
from fog_rl_medical.priority.triage_scorer import TriageScorer
from fog_rl_medical.llm.reasoning_module import ReasoningModule
from fog_rl_medical.llm.prompt_builder import PromptBuilder
from fog_rl_medical.llm.cache import LLMCache
import hashlib
import numpy as np

@dataclass
class PriorityTask:
    task_id: str
    patient_id: str
    priority: int
    deadline: float
    compute_budget: float
    shared_embedding: np.ndarray

class PriorityEngine:
    def __init__(self, llm_config=None, priority_config=None):
        self.scorer = TriageScorer(priority_config)
        self.llm = ReasoningModule(llm_config)
        self.prompt_builder = PromptBuilder()
        self.cache = LLMCache()

    def assign(self, fused_features, task_id="000"):
        patient_id = fused_features.patient_id
        embedding = fused_features.embedding
        
        rule_priority, confidence = self.scorer.score(embedding)
        
        # Check cache
        cache_key = hashlib.md5(embedding[:32].tobytes()).hexdigest()
        cached = self.cache.get(cache_key)
        
        if cached:
            final_priority = cached['priority']
            rationale = cached['rationale']
        else:
            # Build prompt
            prompt = self.prompt_builder.build(
                "vitals mock", "ecg mock", "imaging mock", "text mock", 
                rule_priority, confidence
            )
            # Call LLM reasoning
            llm_res = self.llm.query(prompt)
            print(f"LLM Output for task {task_id}: {llm_res}")
            llm_priority = llm_res['priority']
            llm_conf = llm_res['confidence']
            
            if llm_conf > 0.75:
                final_priority = llm_priority
            else:
                final_priority = rule_priority
                
            rationale = llm_res['rationale']
            self.cache.put(cache_key, {'priority': final_priority, 'rationale': rationale})
            
        # Profiles for deadlines and budgets
        deadlines = {1: 0.1, 2: 0.5, 3: 2.0, 4: 10.0}
        budgets = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}
        
        return PriorityTask(
            task_id=task_id,
            patient_id=patient_id,
            priority=final_priority,
            deadline=deadlines.get(final_priority, 10.0),
            compute_budget=budgets.get(final_priority, 0.2),
            shared_embedding=embedding
        )
