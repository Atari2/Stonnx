# ONNXRustProto

## How to use
- Run `cargo build --release`
- Run `cargo run --release -- --model <modelname>` (running in debug is hyper slow)

## Currently supported models
- mobilenetv2-10 (name to run it is just mobilenet)
- bvlcalexnet-12
- googlenet-12

## Currently work in progress models
- GPT-2 (https://github.com/onnx/models/tree/main/text/machine_comprehension/gpt-2)
- emotion (https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus)
- RFB (https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface)