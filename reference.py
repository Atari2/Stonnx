from onnx.reference.reference_evaluator import ReferenceEvaluator
import numpy as np

sess = ReferenceEvaluator(r'C:\\Users\\VG User\\Documents\\GitHub\\ONNXRustProto\\models\\GPT2\\model.onnx', verbose=5)
results = sess.run(None, {
    'input1': np.random.randn(1, 1, 8).astype(np.int64),
})