from __future__ import annotations
from ctypes import CDLL, c_char_p, c_void_p, c_int64, c_bool, string_at
import sys
from typing import Callable, Optional
from enum import IntEnum

__all__ = ['run_model', 'last_error', 'Verbosity', 'GraphFormat', 'ExecutionMode', 'ONNXModelProto']

_onnxrust_proto = None

MAX_OPSET_VERSION = 20

class ExecutionMode(IntEnum):
    FailFast = 1
    Continue = 0

class GraphFormat(IntEnum):
    NONE = 0
    JSON = 1
    DOT = 2

class Verbosity(IntEnum):
    Minimal = 0
    Informational = 1
    Results = 2
    Intermediate = 4

class _OnnxRustProtoDll:
    def __init__(self, dllname):
        dll = CDLL(dllname)
        self.dll = dll
        self.funcs: dict[str, Callable] = {}
    
    def setup_func(self, name, argtypes, restype):
        func = getattr(self.dll, name)
        func.argtypes = argtypes
        func.restype = restype
        self.funcs[name] = func

def __init_onnxrust_proto_dll():
    global _onnxrust_proto
    if _onnxrust_proto:
        return
    if sys.platform == "win32":
        _onnxrust_proto = _OnnxRustProtoDll("./onnxrust_proto_api.dll")
    elif sys.platform == "darwin":
        _onnxrust_proto = _OnnxRustProtoDll("./libonnxrust_proto_api.dylib")
    else:
        _onnxrust_proto = _OnnxRustProtoDll("./libonnxrust_proto_api.so")

    _onnxrust_proto.setup_func("read_onnx_model", [c_char_p], c_void_p)
    _onnxrust_proto.setup_func("free_onnx_model", [c_void_p], None)
    _onnxrust_proto.setup_func("get_opset_version", [c_void_p], c_int64)
    _onnxrust_proto.setup_func("run_model", [c_char_p, c_int64, c_int64, c_int64], c_bool)


__init_onnxrust_proto_dll()

class ONNXModelProto:
    data_ptr: c_void_p
    freed: bool

    def __init__(self, data_ptr: c_void_p):
        self.data_ptr = data_ptr
        self.freed = False

    @staticmethod
    def from_name(name: str) -> Optional[ONNXModelProto]:
        data_ptr = _onnxrust_proto.funcs["read_onnx_model"](c_char_p(name.encode()))
        if data_ptr == c_void_p():
            return None
        return ONNXModelProto(data_ptr)
    
    def get_opset_version(self) -> int:
        return int(_onnxrust_proto.funcs["get_opset_version"](self.data_ptr))

    def __enter__(self):
        return self
        
    
    def __exit__(self):
        if not self.freed:
            _onnxrust_proto.funcs["free_onnx_model"](self.data_ptr)
            self.freed = True
            self.data_ptr = c_void_p()
    
    def __del__(self):
        if not self.freed:
            _onnxrust_proto.funcs["free_onnx_model"](self.data_ptr)
            self.freed = True
            self.data_ptr = c_void_p()

def last_error() -> str:
    return string_at(_onnxrust_proto.funcs["last_error"]()).decode()

def run_model(mode_name: str, verbosity: Verbosity, graph_format: GraphFormat, failfast: ExecutionMode) -> bool:
    return bool(_onnxrust_proto.funcs["run_model"](c_char_p(mode_name.encode()), c_int64(verbosity.value), c_int64(graph_format.value), c_int64(failfast.value)))
    