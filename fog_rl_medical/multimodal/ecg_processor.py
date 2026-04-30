import numpy as np

class ECGProcessor:
    def process(self, payload):
        # Mock Pan-Tompkins R-peak detection & feature extraction
        # Return 32-dim feature vector
        features = np.random.randn(32)
        return features
