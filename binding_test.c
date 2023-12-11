#include <stdio.h>
#include "bindings/c/onnxrust_proto.h"

int main() {
    int result = run_model("GPT2", VERBOSITY_MINIMAL, GRAPH_FORMAT_NONE, EXECUTION_FAILFAST);
    if (result) {
        printf("Model execution succeeded\n");
        return EXIT_SUCCESS;
    } else {
        printf("Model execution failed\n");
        printf("Error: %s\n", last_error());
        return EXIT_FAILURE;
    }
}