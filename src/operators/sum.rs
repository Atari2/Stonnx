use crate::{
    onnx::NodeProto,
    utils::{ArrayType, BoxResult, OperationResult, ValueType},
};
use anyhow::anyhow;

const _OPSET_VERSIONS: [i64; 4] = [1, 6, 8, 13];

fn _sum_i64(inputs: &[&ArrayType]) -> BoxResult<ArrayType> {
    let mut inputs_i64 = vec![];
    for input in inputs.iter() {
        if let ArrayType::I64(x) = input {
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
    Ok(ArrayType::I64(val))
}

fn _sum_f32(inputs: &[&ArrayType]) -> BoxResult<ArrayType> {
    let mut inputs_f32 = vec![];
    for input in inputs.iter() {
        if let ArrayType::F32(x) = input {
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
    Ok(ArrayType::F32(val))
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_sum.py
/// https://onnx.ai/onnx/operators/onnx__Sum.html
pub fn sum(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
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
