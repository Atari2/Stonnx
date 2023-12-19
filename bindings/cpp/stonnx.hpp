#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

/// Maximum supported opset version
constexpr static const int64_t MAX_OPSET_VERSION = 20;

/// Execution mode
enum class ExecutionMode : int64_t {
  /// Fails immediately if it encounters an operator that is not implemented
  FailFast = 1,
  /// Continues execution if it encounters an operator that is not implemented, simply panicking when it encounters that operator
  Continue = 0,
};

/// Graph format
enum class GraphFormat : int64_t {
  /// No graph output
  None = 0,
  /// Graph output in JSON format, saved to graph.json
  Json = 1,
  /// Graph output in DOT format, saved to graph.dot
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

/// Reads an ONNX model from a file
///
/// Returns a pointer to a model, null if error, check last_error
///
/// # Safety
///
/// Should take a valid path as a C string
ModelProto *read_onnx_model(const char *model_path);

/// Frees a model
///
/// Returns nothing, does nothing if given a null pointer
///
/// # Safety
///
/// Should take a valid pointer to a model
void free_onnx_model(ModelProto *model);

/// Returns the opset version of a model
///
/// Returns MAX_OPSET_VERSION if no opset version is found
///
/// Returns MAX_OPSET_VERSION if given a null pointer and sets last_error
///
/// # Safety
///
/// Should take a valid pointer to a model
int64_t get_opset_version(const ModelProto *model);

/// Runs a model given a path to a model directory, verbosity level (0-4), graph format (json / dot), and execution mode (failfast / continue)
///
/// Returns true if successful, false if not
///
/// Sets last_error if an error occurs
///
/// # Safety
///
/// Should take a valid path to a model directory as a C string
///
/// Should take a valid verbosity level
///
/// Should take a valid graph format
///
/// Should take a valid execution mode
bool run_model(const char *model_path,
               Verbosity verbosity,
               GraphFormat graph_format,
               ExecutionMode failfast);

/// Returns a pointer to a C string containing the last error
///
/// Returns a null pointer if no error is present
///
/// # Safety
///
/// Safe, returns a pointer to a C string, null if no error
///
/// Valid until the next call to run_model
const char *last_error();

} // extern "C"
