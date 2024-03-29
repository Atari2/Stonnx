#![allow(dead_code)]
pub mod common;
pub mod executor;
pub mod onnxparser;
pub mod operators;
pub mod parallel;
pub mod protograph;
pub mod utils;

use crate::common::MAX_OPSET_VERSION;
use crate::executor::execute_model;
use crate::onnxparser::onnx;
use crate::utils::read_model;
use common::{Args, BoxResult, VerbosityLevel};
use once_cell::sync::Lazy;
use utils::OutputInfo;
use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};

static mut LAST_ERROR: Lazy<CString> = Lazy::new(|| CString::new(*b"").unwrap());

#[repr(i64)]
pub enum Verbosity {
    Silent = -1,
    Minimal = 0,
    Informational = 1,
    Results = 2,
    Intermediate = 4,
}

#[repr(i64)]
#[derive(PartialEq)]
/// Graph format
pub enum GraphFormat {
    /// No graph output
    None = 0,
    /// Graph output in JSON format, saved to graph.json
    Json = 1,
    /// Graph output in DOT format, saved to graph.dot
    Dot = 2,
}

#[repr(i64)]
#[derive(PartialEq)]
/// Execution mode
pub enum ExecutionMode {
    /// Fails immediately if it encounters an operator that is not implemented
    FailFast = 1,
    /// Continues execution if it encounters an operator that is not implemented, simply panicking when it encounters that operator
    Continue = 0,
}

#[no_mangle]
/// Reads an ONNX model from a file
///
/// Returns a pointer to a model, null if error, check last_error
///
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
/// Frees a model
///
/// Returns nothing, does nothing if given a null pointer
///
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
/// Returns the opset version of a model
///
/// Returns MAX_OPSET_VERSION if no opset version is found
///
/// Returns MAX_OPSET_VERSION if given a null pointer and sets last_error
///
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

pub struct Model {
    path: PathBuf,
    verbose: VerbosityLevel,
    graph: bool,
    graph_format: String,
    failfast: bool,
} 

impl Model {
    pub fn path(&mut self, path: &str) -> &mut Self {
        self.path = PathBuf::from(path);
        self
    }
    pub fn verbose(&mut self, verbose: VerbosityLevel) -> &mut Self {
        self.verbose = verbose;
        self
    }
    pub fn graph(&mut self, graph: bool) -> &mut Self {
        self.graph = graph;
        self
    }
    pub fn graph_format(&mut self, graph_format: &str) -> &mut Self {
        self.graph_format = graph_format.to_owned();
        self
    }
    pub fn run(&self) -> BoxResult<HashMap<String, OutputInfo>> {
        let args = Args::new(
            self.path.clone(),
            self.verbose,
            self.graph,
            self.graph_format.clone(),
            self.failfast,
        );
        crate::execute_model(&args)
    }
}

impl Default for Model {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            verbose: VerbosityLevel::Minimal,
            graph: false,
            graph_format: "".to_owned(),
            failfast: false,
        }
    }
}

#[no_mangle]
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
        verbosity as i64,
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
/// Returns a pointer to a C string containing the last error
///
/// Returns a null pointer if no error is present
///
/// # Safety
///
/// Safe, returns a pointer to a C string, null if no error
///
/// Valid until the next call to run_model
pub unsafe extern "C" fn last_error() -> *const std::os::raw::c_char {
    if LAST_ERROR.is_empty() {
        return std::ptr::null();
    }
    LAST_ERROR.as_ptr()
}
