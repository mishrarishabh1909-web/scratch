import numpy as np
import time

class WorkloadGenerator:
    def __init__(self, config=None):
        self.config = config or {}
        np.random.seed(42)

    def generate_ecg_stream(self, patient_id, condition="normal"):
        # Synthetic ECG at 250Hz for 10s (2500 samples)
        base_signal = np.sin(np.linspace(0, 10 * np.pi, 2500))
        if condition == "arrhythmia":
            base_signal += np.random.normal(0, 0.5, 2500)
        return {
            'patient_id': patient_id,
            'modality': 'ECG',
            'raw_payload': base_signal,
            'timestamp': time.time()
        }

    def generate_vitals(self, patient_id, severity="low"):
        # Random parameters corresponding to NEWS2 scoring components
        vitals = {
            'RR': np.random.randint(12, 20) if severity == "low" else np.random.randint(25, 35),
            'SpO2': np.random.randint(96, 100) if severity == "low" else np.random.randint(85, 92),
            'temp': np.round(np.random.uniform(36.5, 37.5), 1),
            'BP_systolic': np.random.randint(110, 130) if severity == "low" else np.random.randint(80, 100),
            'HR': np.random.randint(60, 90) if severity == "low" else np.random.randint(110, 140),
            'AVPU': 'A' if severity == "low" else 'V'
        }
        return {
            'patient_id': patient_id,
            'modality': 'VITALS',
            'raw_payload': vitals,
            'timestamp': time.time()
        }

    def generate_image_task(self, patient_id, modality="X-RAY", urgency=1):
        # Generate dummy 3-channel 224x224 image
        image_data = np.random.rand(3, 224, 224).astype(np.float32)
        return {
            'patient_id': patient_id,
            'modality': 'IMAGE',
            'image_type': modality,
            'raw_payload': image_data,
            'urgency': urgency,
            'timestamp': time.time()
        }

    def generate_clinical_note(self, patient_id, keywords=None):
        keywords = keywords or ["stable", "routine check"]
        note = f"Patient {patient_id} clinical evaluation. Note: " + " ".join(keywords)
        return {
            'patient_id': patient_id,
            'modality': 'TEXT',
            'raw_payload': note,
            'timestamp': time.time()
        }
