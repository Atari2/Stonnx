#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define VERBOSITY_MINIMAL 0

#define VERBOSITY_INFORMATIONAL 1

#define VERBOSITY_RESULTS 2

#define VERBOSITY_INTERMEDIATE 4

#define GRAPH_FORMAT_NONE 0

#define GRAPH_FORMAT_JSON 1

#define GRAPH_FORMAT_DOT 2

#define EXECUTION_FAILFAST 1

#define EXECUTION_CONTINUE 0

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

/**
 * # Safety
 *
 * Should take a valid path to a model directory as a C string
 */
int run_model(const char *model_path, int verbosity, int graph_format, bool failfast);

/**
 * # Safety
 *
 * Safe, returns a pointer to a C string, null if no error
 * Valid until the next call to run_model
 */
const char *last_error(void);
