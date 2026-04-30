import numpy as np

class ImagingProcessor:
    def process(self, payload):
        # Mock ONNX MobileNetV3 small inference
        # Return 128-dim feature vector
        features = np.random.randn(128)
        return features
