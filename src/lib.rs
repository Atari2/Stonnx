#![allow(dead_code)]
mod common;
mod onnxparser;
mod utils;
mod executor;
mod operators;
mod protograph;

use std::path::Path;
use common::Args;
use once_cell::sync::OnceCell;
use crate::executor::execute_model;
use crate::common::MAX_OPSET_VERSION;
use crate::onnxparser::onnx;
use crate::utils::read_model;

static LAST_ERROR: OnceCell<Box<std::ffi::CString>> = OnceCell::new();

pub const VERBOSITY_MINIMAL: std::os::raw::c_int = 0;
pub const VERBOSITY_INFORMATIONAL: std::os::raw::c_int = 1;
pub const VERBOSITY_RESULTS: std::os::raw::c_int = 2;
pub const VERBOSITY_INTERMEDIATE: std::os::raw::c_int = 4;
pub const GRAPH_FORMAT_NONE: std::os::raw::c_int = 0;
pub const GRAPH_FORMAT_JSON: std::os::raw::c_int = 1;
pub const GRAPH_FORMAT_DOT: std::os::raw::c_int = 2;
pub const EXECUTION_FAILFAST: std::os::raw::c_int = 1;
pub const EXECUTION_CONTINUE: std::os::raw::c_int = 0;

#[no_mangle]
/// # Safety
/// 
/// Should take a valid path as a C string
pub unsafe extern "C" fn read_onnx_model(model_path: *const std::os::raw::c_char) -> *mut onnx::ModelProto {
    let model_path = unsafe { std::ffi::CStr::from_ptr(model_path) };
    let model_path = model_path.to_str().unwrap();
    let model = read_model(Path::new(model_path)).unwrap();
    Box::into_raw(Box::new(model))
}

#[no_mangle]
/// # Safety
/// 
/// Should take a valid pointer to a model
pub unsafe extern "C" fn free_onnx_model(model: *mut onnx::ModelProto) {
    unsafe {
        drop(Box::from_raw(model));
    }
}

#[no_mangle]
/// # Safety
/// 
/// Should take a valid pointer to a model
pub unsafe extern "C" fn get_opset_version(model: *const onnx::ModelProto) -> i64 {
    unsafe {
        if let Some(v) = (*model).opset_import.get(0) {
            if let Some(v) = v.version {
                v
            } else {
                MAX_OPSET_VERSION
            }
        } else {
            MAX_OPSET_VERSION
        }
    }
}

#[no_mangle]
/// # Safety
/// 
/// Should take a valid path to a model directory as a C string
pub unsafe extern "C" fn run_model(
    model_path: *const std::os::raw::c_char, 
    verbosity: std::os::raw::c_int,
    graph_format: std::os::raw::c_int,
    failfast: bool
) -> std::os::raw::c_int {
    let model_path = unsafe { std::ffi::CStr::from_ptr(model_path) };
    let model_path = model_path.to_str().unwrap();
    let gf = match graph_format {
        GRAPH_FORMAT_NONE => std::ffi::CStr::from_ptr(b"\0" as *const u8 as *const i8),
        GRAPH_FORMAT_JSON => std::ffi::CStr::from_ptr(b"json\0" as *const u8 as *const i8),
        GRAPH_FORMAT_DOT => std::ffi::CStr::from_ptr(b"dot\0" as *const u8 as *const i8),
        _ => {
            LAST_ERROR.set(Box::new(std::ffi::CString::new("Invalid graph format").unwrap())).unwrap();
            return 0;
        }
    };
    let args = Args::from_parts(
        model_path.into(),
        verbosity as u64,
        graph_format != GRAPH_FORMAT_NONE,
        gf.to_owned().into_string().unwrap(),
        failfast
    );
    match crate::execute_model(&args) {
        Ok(_) => 1,
        Err(e) => {
            LAST_ERROR.set(Box::new(std::ffi::CString::new(e.to_string()).unwrap())).unwrap();
            0
        }
    }
}

#[no_mangle]
/// # Safety
/// 
/// Safe, returns a pointer to a C string, null if no error
/// Valid until the next call to run_model
pub unsafe extern "C" fn last_error() -> *const std::os::raw::c_char {
    LAST_ERROR.get().map_or(std::ptr::null(), |s| s.as_ptr())
}