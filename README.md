# ONNXRustProto

## How to use
- Run `cargo build --release`
- Run `cargo run --release -- --model <modelname>` (running in debug is hyper slow)

## Currently supported models
- mobilenetv2-10 (name to run it is just mobilenet) with 0 differences larger than 10e-5
- bvlcalexnet-12 with 284 differences by more than 10e-5
- googlenet-12 with 353 differences by more than 10e-5
- GPT-2 with 0 differences larger than 10e-5
- RFB (https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface) with 68040+24192 differences by more than 10e-5
- emotion (https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus) with 0 differences larger than 10e-5
