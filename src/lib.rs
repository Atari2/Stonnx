#![allow(dead_code)]
mod common;
mod executor;
mod onnxparser;
mod operators;
mod parallel;
mod protograph;
mod utils;

use crate::common::MAX_OPSET_VERSION;
use crate::executor::execute_model;
use crate::onnxparser::onnx;
use crate::utils::read_model;
use common::Args;
use once_cell::sync::Lazy;
use std::ffi::CString;
use std::path::Path;

static mut LAST_ERROR: Lazy<CString> = Lazy::new(|| CString::new(b"").unwrap());

#[repr(i64)]
pub enum Verbosity {
    Minimal = 0,
    Informational = 1,
    Results = 2,
    Intermediate = 4,
}

#[repr(i64)]
#[derive(PartialEq)]
pub enum GraphFormat {
    None = 0,
    Json = 1,
    Dot = 2,
}

#[repr(i64)]
#[derive(PartialEq)]
pub enum ExecutionMode {
    FailFast = 1,
    Continue = 0,
}

#[no_mangle]
/// # Safety
///
/// Should take a valid path as a C string
pub unsafe extern "C" fn read_onnx_model(
    model_path: *const std::os::raw::c_char,
) -> *mut onnx::ModelProto {
    let model_path = unsafe { std::ffi::CStr::from_ptr(model_path) };
    let model_path = match model_path.to_str() {
        Ok(s) => s,
        Err(e) => {
            let e = match CString::new(e.to_string()) {
                Ok(s) => s,
                Err(_) => {
                    return std::ptr::null_mut();
                }
            };
            *LAST_ERROR = e;
            return std::ptr::null_mut();
        }
    };
    let model = match read_model(Path::new(model_path)) {
        Ok(m) => m,
        Err(e) => {
            let e = match CString::new(e.to_string()) {
                Ok(s) => s,
                Err(_) => {
                    return std::ptr::null_mut();
                }
            };
            *LAST_ERROR = e;
            return std::ptr::null_mut();
        }
    };
    Box::into_raw(Box::new(model))
}

#[no_mangle]
/// # Safety
///
/// Should take a valid pointer to a model
pub unsafe extern "C" fn free_onnx_model(model: *mut onnx::ModelProto) {
    if model.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(model));
    }
}

#[no_mangle]
/// # Safety
///
/// Should take a valid pointer to a model
pub unsafe extern "C" fn get_opset_version(model: *const onnx::ModelProto) -> i64 {
    if model.is_null() {
        *LAST_ERROR = CString::new("NULL pointer passed to get_opset_version").unwrap();
        return MAX_OPSET_VERSION;
    }
    unsafe {
        if let Some(v) = (*model).opset_import.first() {
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
/// Should take a valid verbosity level
/// Should take a valid graph format
/// Should take a valid execution mode
pub unsafe extern "C" fn run_model(
    model_path: *const std::os::raw::c_char,
    verbosity: Verbosity,
    graph_format: GraphFormat,
    failfast: ExecutionMode,
) -> bool {
    let model_path = unsafe { std::ffi::CStr::from_ptr(model_path) };
    let model_path = match model_path.to_str() {
        Ok(s) => s,
        Err(e) => {
            let e = match CString::new(e.to_string()) {
                Ok(s) => s,
                Err(_) => {
                    return false;
                }
            };
            *LAST_ERROR = e;
            return false;
        }
    };
    let gf = match graph_format {
        GraphFormat::None => "".to_owned(),
        GraphFormat::Json => "json".to_owned(),
        GraphFormat::Dot => "dot".to_owned(),
    };
    let args = Args::from_parts(
        model_path.into(),
        verbosity as u64,
        graph_format != GraphFormat::None,
        gf,
        failfast != ExecutionMode::Continue,
    );
    match crate::execute_model(&args) {
        Ok(_) => true,
        Err(e) => {
            let e = match CString::new(e.to_string()) {
                Ok(s) => s,
                Err(_) => {
                    return false;
                }
            };
            *LAST_ERROR = e;
            false
        }
    }
}

#[no_mangle]
/// # Safety
///
/// Safe, returns a pointer to a C string, null if no error
/// Valid until the next call to run_model
pub unsafe extern "C" fn last_error() -> *const std::os::raw::c_char {
    LAST_ERROR.as_ptr()
}
