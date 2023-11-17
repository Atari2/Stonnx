use ndarray::{ArrayD, Axis, CowArray, IxDyn};

use crate::{
    onnx::NodeProto,
    utils::{ArrayType, BoxResult},
    utils::{OperationResult, ValueType},
};

const _OPSET_VERSIONS: [i64; 4] = [1, 4, 11, 13];

#[derive(Debug)]
struct ConcatAttrs {
    axis: i64,
}

impl ConcatAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            axis: node
                .attribute
                .iter()
                .find(|a| a.name() == "axis")
                .map_or(1, |a| a.i.unwrap_or(1)),
        }
    }
}

fn _preprocess<A: Clone>(a: &ArrayD<A>, axis: i64) -> BoxResult<CowArray<'_, A, IxDyn>> {
    if a.shape().is_empty() {
        return Err("Input must be at least 1D".into());
    }
    if axis >= a.shape().len() as i64 {
        let axis = axis as usize;
        let new_shape = a
            .shape()
            .iter()
            .chain(std::iter::repeat(&1).take(axis + 1 - a.shape().len()))
            .copied()
            .collect::<Vec<_>>();
        Ok(a.to_shape(new_shape)?.into_dyn()) // can fail is array not contiguous
    } else {
        Ok(a.into())
    }
}

fn _concat_i64(inputs: &[&ArrayType], attrs: ConcatAttrs) -> BoxResult<ArrayType> {
    let mut inputs_i64 = vec![];
    let mut cow_array = vec![];
    for input in inputs.iter() {
        if let ArrayType::I64(x) = input {
            cow_array.push(_preprocess(x, attrs.axis)?);
        } else {
            return Err("Inputs not all of I64".into());
        }
    }
    for array in cow_array.iter() {
        inputs_i64.push(array.view());
    }
    if inputs_i64.is_empty() {
        return Err("No inputs".into());
    }
    let shape_i64 = inputs_i64[0].shape();
    let axis = if attrs.axis < 0 {
        shape_i64.len() as i64 + attrs.axis
    } else {
        attrs.axis
    } as usize;
    Ok(ArrayType::I64(ndarray::concatenate(
        Axis(axis),
        &inputs_i64,
    )?))
}

fn _concat_f32(inputs: &[&ArrayType], attrs: ConcatAttrs) -> BoxResult<ArrayType> {
    let mut inputs_f32 = vec![];
    let mut cow_array = vec![];
    for input in inputs.iter() {
        if let ArrayType::F32(x) = input {
            cow_array.push(_preprocess(x, attrs.axis)?);
        } else {
            return Err("Inputs not all of F32".into());
        }
    }
    for array in cow_array.iter() {
        inputs_f32.push(array.view());
    }
    if inputs_f32.is_empty() {
        return Err("No inputs".into());
    }
    let shape_f32 = inputs_f32[0].shape();
    let axis = if attrs.axis < 0 {
        shape_f32.len() as i64 + attrs.axis
    } else {
        attrs.axis
    } as usize;
    Ok(ArrayType::F32(ndarray::concatenate(
        Axis(axis),
        &inputs_f32,
    )?))
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_concat.py
/// https://onnx.ai/onnx/operators/onnx__Concat.html
pub fn concat(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let attrs = ConcatAttrs::new(node);

    if inputs.is_empty() {
        return Err("No inputs".into());
    }

    let type_ = inputs[0].value_type();

    match type_ {
        ValueType::I64 => Ok(_concat_i64(inputs, attrs)?.into()),
        ValueType::F32 => Ok(_concat_f32(inputs, attrs)?.into()),
        _ => Err("Only f32 and i64 are supported".into()),
    }
}
