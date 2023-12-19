#include <stdio.h>
#include "../c/stonnx.h"

int main()
{
    bool ok = run_model("GPT2", Minimal, None, FailFast);
    if (ok)
    {
        printf("Model execution succeeded\n");
        return EXIT_SUCCESS;
    }
    else
    {
        printf("Model execution failed\n");
        printf("Error: %s\n", last_error());
        return EXIT_FAILURE;
    }
}