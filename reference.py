from onnx.reference.reference_evaluator import ReferenceEvaluator
import onnx
import numpy as np
import argparse
import json
from onnx import numpy_helper

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str)
parser.add_argument('-v', '--verbose', type=int, default=4)
args = parser.parse_args()

jsonfile = f'models/{args.model}/inputs.json'

with open(jsonfile, 'r') as f:
    information = json.load(f)

modelpath = f'models/{args.model}/{information["modelpath"]}'

inputpaths = [f'models/{args.model}/{path}' for path in information['inputs']]
outputpaths = [f'models/{args.model}/{path}' for path in information['outputs']]

model = onnx.load_model(modelpath)
tensor_inputs = [onnx.load_tensor(path) for path in inputpaths]
tensor_outputs = [onnx.load_tensor(path) for path in outputpaths]

inputs = {}
for input in tensor_inputs:
    inputs[input.name] = numpy_helper.to_array(input)


sess = ReferenceEvaluator(model, verbose=args.verbose)
results = sess.run(None, inputs)
if len(results) != len(tensor_outputs):
    raise Exception("Number of outputs do not match")


for (res, tensor) in zip(results, tensor_outputs):
    tensor_arr = numpy_helper.to_array(tensor)
    np.testing.assert_almost_equal(res, tensor_arr, decimal=5)