#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

/// Maximum supported opset version
constexpr static const int64_t MAX_OPSET_VERSION = 20;

enum class ExecutionMode : int64_t {
  FailFast = 1,
  Continue = 0,
};

enum class GraphFormat : int64_t {
  None = 0,
  Json = 1,
  Dot = 2,
};

enum class Verbosity : int64_t {
  Minimal = 0,
  Informational = 1,
  Results = 2,
  Intermediate = 4,
};

struct ModelProto;

extern "C" {

/// # Safety
///
/// Should take a valid path as a C string
ModelProto *read_onnx_model(const char *model_path);

/// # Safety
///
/// Should take a valid pointer to a model
void free_onnx_model(ModelProto *model);

/// # Safety
///
/// Should take a valid pointer to a model
int64_t get_opset_version(const ModelProto *model);

/// # Safety
///
/// Should take a valid path to a model directory as a C string
/// Should take a valid verbosity level
/// Should take a valid graph format
/// Should take a valid execution mode
bool run_model(const char *model_path,
               Verbosity verbosity,
               GraphFormat graph_format,
               ExecutionMode failfast);

/// # Safety
///
/// Safe, returns a pointer to a C string, null if no error
/// Valid until the next call to run_model
const char *last_error();

} // extern "C"
