import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

@dataclass
class FusedFeatures:
    embedding: np.ndarray
    modality_mask: np.ndarray
    patient_id: str

class FusionEngine(nn.Module):
    def __init__(self):
        super(FusionEngine, self).__init__()
        # Total dims: 32 (ECG) + 16 (VITALS) + 128 (IMAGE) + 64 (TEXT) = 240
        self.mlp = nn.Sequential(
            nn.Linear(240, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def process_multimodal(self, task_dict_list, patient_id):
        ecg = np.zeros(32)
        vitals = np.zeros(16)
        image = np.zeros(128)
        text = np.zeros(64)
        mask = np.zeros(4)
        
        for t in task_dict_list:
            mod = t.get('tagged_modality')
            features = t.get('features')
            if features is None:
                continue
                
            if mod == 'ECG':
                ecg = features
                mask[0] = 1
            elif mod == 'VITALS':
                vitals = features
                mask[1] = 1
            elif mod == 'IMAGE':
                image = features
                mask[2] = 1
            elif mod == 'TEXT':
                text = features
                mask[3] = 1
                
        concat_features = np.concatenate([ecg, vitals, image, text])
        tensor_feats = torch.tensor(concat_features, dtype=torch.float32)
        
        with torch.no_grad():
            shared_embedding = self.mlp(tensor_feats).numpy()
            
        return FusedFeatures(
            embedding=shared_embedding,
            modality_mask=mask,
            patient_id=patient_id
        )
