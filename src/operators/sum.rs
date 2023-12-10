use crate::{
    common::{BoxResult, OperatorResult, TensorType, ValueType},
    onnx::NodeProto,
};
use anyhow::anyhow;

const _OPSET_VERSIONS: [i64; 4] = [1, 6, 8, 13];

fn _sum_i64(inputs: &[&TensorType]) -> BoxResult<TensorType> {
    let mut inputs_i64 = vec![];
    for input in inputs.iter() {
        if let TensorType::I64(x) = input {
            inputs_i64.push(x.view());
        } else {
            return Err(anyhow!("Inputs not all of I64"));
        }
    }
    if inputs_i64.is_empty() {
        return Err(anyhow!("No inputs"));
    }
    let val = inputs_i64
        .iter()
        .skip(1)
        .fold(inputs_i64[0].to_owned(), |acc, x| acc + x);
    Ok(TensorType::I64(val))
}

fn _sum_f32(inputs: &[&TensorType]) -> BoxResult<TensorType> {
    let mut inputs_f32 = vec![];
    for input in inputs.iter() {
        if let TensorType::F32(x) = input {
            inputs_f32.push(x.view());
        } else {
            return Err(anyhow!("Inputs not all of I64"));
        }
    }
    if inputs_f32.is_empty() {
        return Err(anyhow!("No inputs"));
    }
    let val = inputs_f32
        .iter()
        .skip(1)
        .fold(inputs_f32[0].to_owned(), |acc, x| acc + x);
    Ok(TensorType::F32(val))
}

/// Element-wise sum of each of the input tensors.
///
/// All inputs and outputs must have the same data type.
///
/// [Python reference](<https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_sum.py>)
///
/// [ONNX Documentation](<https://onnx.ai/onnx/operators/onnx__Sum.html>)
pub fn sum(
    inputs: &[&TensorType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    if inputs.is_empty() {
        return Err(anyhow!("No inputs"));
    }

    let type_ = inputs[0].value_type();

    match type_ {
        ValueType::I64 => Ok(_sum_i64(inputs)?.into()),
        ValueType::F32 => Ok(_sum_f32(inputs)?.into()),
        _ => Err(anyhow!("Only f32 and i64 are supported")),
    }
}
