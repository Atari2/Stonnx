#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Maximum supported opset version
 */
#define MAX_OPSET_VERSION 20

enum ExecutionMode {
  FailFast = 1,
  Continue = 0,
};
typedef int64_t ExecutionMode;

enum GraphFormat {
  None = 0,
  Json = 1,
  Dot = 2,
};
typedef int64_t GraphFormat;

enum Verbosity {
  Minimal = 0,
  Informational = 1,
  Results = 2,
  Intermediate = 4,
};
typedef int64_t Verbosity;

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

/**
 * # Safety
 *
 * Should take a valid path to a model directory as a C string
 * Should take a valid verbosity level
 * Should take a valid graph format
 * Should take a valid execution mode
 */
bool run_model(const char *model_path,
               Verbosity verbosity,
               GraphFormat graph_format,
               ExecutionMode failfast);

/**
 * # Safety
 *
 * Safe, returns a pointer to a C string, null if no error
 * Valid until the next call to run_model
 */
const char *last_error(void);
