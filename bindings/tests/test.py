import onnxrust_proto

ok = onnxrust_proto.run_model("GPT2", 
                              onnxrust_proto.Verbosity.Minimal,
                              onnxrust_proto.GraphFormat.NONE,
                              onnxrust_proto.ExecutionMode.FailFast)
if ok:
    print("Model execution succeeded")
    exit(0)
else:
    print(f"Model execution failed\nError: {onnxrust_proto.last_error()}")
    exit(1)