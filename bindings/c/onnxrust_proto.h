#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Maximum supported opset version
 */
#define MAX_OPSET_VERSION 20

typedef struct ModelProto ModelProto;

/**
 * # Safety
 *
 * Should take a valid path as a C string
 */
struct ModelProto *read_onnx_model(const char *model_path);

/**
 * # Safety
 *
 * Should take a valid pointer to a model
 */
void free_onnx_model(struct ModelProto *model);

/**
 * # Safety
 *
 * Should take a valid pointer to a model
 */
int64_t get_opset_version(const struct ModelProto *model);
