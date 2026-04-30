class PromptBuilder:
    def build(self, vitals_summary, ecg_summary, imaging_summary, text_excerpt, rule_priority, confidence):
        prompt = f"""Patient data summary: {vitals_summary}
ECG findings: {ecg_summary}
Imaging: {imaging_summary}
Clinical notes: {text_excerpt}
Rule-based triage: Priority {rule_priority}
Confidence: {confidence:.2f}
Question: Should this case be escalated or maintained?
Respond in JSON: {{"priority": int, "rationale": "string", "confidence": float}}"""
        return prompt
