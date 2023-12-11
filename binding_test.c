#include <stdio.h>
#include "bindings/c/onnxrust_proto.h"

int main() {
    const char *model_path = "models/GPT2/model.onnx";
    ModelProto *model = read_onnx_model(model_path);
    printf("Model opset version: %lld\n", get_opset_version(model));
    free_onnx_model(model);
}