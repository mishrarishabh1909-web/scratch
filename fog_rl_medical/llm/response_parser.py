import json

class ResponseParser:
    def parse(self, raw_response):
        try:
            if isinstance(raw_response, str):
                data = json.loads(raw_response)
            else:
                data = raw_response
                
            priority = int(data.get('priority', 4))
            rationale = str(data.get('rationale', ''))
            confidence = float(data.get('confidence', 0.0))
            
            # Clamp priority to 1-4
            priority = max(1, min(4, priority))
            
            return priority, rationale, confidence
        except Exception:
            return 4, "Failed to parse", 0.0
