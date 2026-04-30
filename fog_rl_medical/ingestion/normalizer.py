import numpy as np

class Normalizer:
    def normalize(self, task):
        modality = task.get('tagged_modality')
        payload = task.get('raw_payload')
        
        if modality == 'ECG':
            # Mock z-score normalization
            if isinstance(payload, np.ndarray):
                mean = np.mean(payload)
                std = np.std(payload)
                if std > 0:
                    task['normalized_payload'] = (payload - mean) / std
                else:
                    task['normalized_payload'] = payload - mean
        
        elif modality == 'VITALS' and isinstance(payload, dict):
            # Mock min-max scaling [0, 1]
            task['normalized_payload'] = {k: float(v) / 200.0 if isinstance(v, (int, float)) else v for k, v in payload.items()}
            
        elif modality == 'IMAGE':
            # Mock ImageNet normalization
            if isinstance(payload, np.ndarray):
                task['normalized_payload'] = (payload - 0.5) / 0.5
            
        elif modality == 'TEXT':
            # Mock text processing
            if isinstance(payload, str):
                text = payload.lower()
                text = text.replace("[phi]", "")
                task['normalized_payload'] = text[:512]
            
        return task
