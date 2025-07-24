import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
import joblib


class ONNXWrapper:
    def __init__(self, sklearn_model, model_name, cache_dir="model/onnx_cache"):
        self.model = sklearn_model
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.session = None
        self.input_names = None
        self.output_name = None
        self._initialize()

    def _initialize(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        onnx_path = os.path.join(self.cache_dir, f"{self.model_name}.onnx")
        input_path = os.path.join(self.cache_dir, f"{self.model_name}_input_names.pkl")

        if os.path.exists(onnx_path):
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.input_names = joblib.load(input_path)
            self.output_name = self.session.get_outputs()[0].name
        else:
            # Convert to ONNX
            X_dummy = np.random.randn(1, len(self.model.feature_names_in_)).astype(np.float32)
            initial_type = [("input", FloatTensorType([None, X_dummy.shape[1]]))]
            onnx_model = convert_sklearn(self.model, initial_types=initial_type)
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.input_names = self.model.feature_names_in_
            self.output_name = self.session.get_outputs()[0].name
            joblib.dump(self.input_names, input_path)

    def predict(self, x_dict):
        x_array = np.array([[x_dict.get(name, 0.0) for name in self.input_names]], dtype=np.float32)
        ort_inputs = {self.session.get_inputs()[0].name: x_array}
        pred = self.session.run([self.output_name], ort_inputs)[0]
        return float(pred[0][0])
