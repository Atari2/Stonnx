import stonnx

ok = stonnx.run_model("GPT2", 
                        stonnx.Verbosity.Minimal,
                        stonnx.GraphFormat.NONE,
                        stonnx.ExecutionMode.FailFast)
if ok:
    print("Model execution succeeded")
    exit(0)
else:
    print(f"Model execution failed\nError: {stonnx.last_error()}")
    exit(1)