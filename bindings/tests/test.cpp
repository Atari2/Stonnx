#include "../cpp/stonnx.hpp"
#include <iostream>

int main() {
    bool ok = run_model("GPT2", Verbosity::Minimal, GraphFormat::None, ExecutionMode::FailFast);
    if (ok) {
        std::cout << "Model execution succeeded\n";
        return EXIT_SUCCESS;
    } else {
        std::cout << "Model execution failed\n";
        std::cout << "Error: " << last_error() << '\n';
        return EXIT_FAILURE;
    }
}