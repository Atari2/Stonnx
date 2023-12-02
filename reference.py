from onnx.reference.reference_evaluator import ReferenceEvaluator
import onnx
import numpy as np
import argparse
import json
from onnx import numpy_helper
import os
import functools
import glob
from pathlib import Path
from onnx.reference.ops.op_average_pool import AveragePool_11
from onnx.reference.ops.op_max_pool import MaxPool

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str)
parser.add_argument("-v", "--verbose", type=int, default=4)
parser.add_argument("-f", "--on-failure", type=str, default="terminate")
args = parser.parse_args()

pmodel = Path(args.model)
if not pmodel.is_absolute():
    jsonfile = f"models/{args.model}/inputs.json"
else:
    jsonfile = pmodel.joinpath("inputs.json")

with open(jsonfile, "r") as f:
    information = json.load(f)

modelpath = (
    f'models/{args.model}/{information["modelpath"]}'
    if not os.path.isabs(information["modelpath"])
    else information["modelpath"]
)

inputpaths = [
    f"models/{args.model}/{path}" if not os.path.isabs(path) else path
    for path in information["inputs"]
]
outputpaths = [
    f"models/{args.model}/{path}" if not os.path.isabs(path) else path
    for path in information["outputs"]
]

model = onnx.load_model(modelpath)
tensor_inputs = [onnx.load_tensor(path) for path in inputpaths]
tensor_outputs = [onnx.load_tensor(path) for path in outputpaths]

inputs = {}
for input in tensor_inputs:
    inputs[input.name] = numpy_helper.to_array(input)

sess = ReferenceEvaluator(model, verbose=4)

modelname = args.model if not pmodel.is_absolute() else pmodel.stem

os.makedirs(f"outputs\\reference\\{modelname}", exist_ok=True)

output_info = {}

output_exit_order = []


def _run(og_run, *op_args, **kwargs):
    outputs = og_run(*op_args, **kwargs)
    if args.verbose > 0:
        output_names = [o.replace("/", "_") for o in og_run.__self__.output]
        for o in output_names:
            output_info[o] = (og_run.__self__.op_type, og_run.__self__.input)
            output_exit_order.append(o)
        for name, output in zip(output_names, outputs):
            np.save(f"outputs\\reference\\{modelname}\\{name}.npy", output)
    return outputs


for node in sess.rt_nodes_:
    og_run = node._run
    node._run = functools.partial(_run, og_run)
results = sess.run(None, inputs)
if len(results) != len(tensor_outputs):
    raise Exception("Number of outputs do not match")

if args.verbose > 0:
    reference_outputs = {}
    rust_outputs = {}
    reference_output_files = glob.glob(f"outputs\\reference\\{modelname}\\*.npy")
    rust_output_files = glob.glob(f"outputs\\{modelname}\\*.npy")

    for file in reference_output_files:
        reference_outputs[os.path.basename(file)] = np.load(file)

    for file in rust_output_files:
        rust_outputs[os.path.basename(file)] = np.load(file)

    for o in output_exit_order:
        key = f"{o}.npy"
        exp = reference_outputs[key]
        act = rust_outputs[key]
        output_name = key.split(".")[0]
        op_type, input = output_info[output_name]
        try:
            np.testing.assert_almost_equal(
                exp,
                act,
                decimal=5,
                verbose=True,
                err_msg=f"Output {key} does not match for operator {op_type} with input {input}",
            )
            print(f"Output {key} matches for operator {op_type} with input {input}")
        except AssertionError as e:
            print(str(e))
            if args.on_failure == "terminate":
                exit(1)
