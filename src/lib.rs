#![allow(dead_code)]
mod common;
mod onnxparser;
mod utils;

use std::path::Path;
use crate::common::MAX_OPSET_VERSION;
use crate::onnxparser::onnx;
use crate::utils::read_model;

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