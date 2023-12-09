use crate::{
    common::{ArrayType, BoxResult, OperationResult},
    onnx::NodeProto,
};
use anyhow::anyhow;
use ndarray::ArrayD;

const _OPSET_VERSIONS: [i64; 4] = [1, 9, 11, 13];

#[derive(Debug)]
struct FlattenAttrs {
    axis: i64,
}

impl FlattenAttrs {
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

fn _flatten<D: Clone>(input: &ArrayD<D>, axis: i64) -> BoxResult<ArrayD<D>> {
    let shape = input.shape();
    let new_shape = if axis == 0 {
        vec![1, shape.iter().product()]
    } else {
        let (left, right) = shape.split_at(axis as usize);
        vec![left.iter().product(), right.iter().product()]
    };
    Ok(input.to_owned().into_shape(new_shape)?)
}

/// <https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_flatten.py>
/// <https://onnx.ai/onnx/operators/onnx__Flatten.html>
pub fn flatten(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let attrs = FlattenAttrs::new(node);
    let input = inputs.get(0).ok_or(anyhow!("No input"))?;
    let axis = if attrs.axis < 0 {
        input.ndim() as i64 + attrs.axis
    } else {
        attrs.axis
    };
    match input {
        ArrayType::F32(x) => Ok(ArrayType::F32(_flatten(x, axis)?).into()),
        ArrayType::I64(x) => Ok(ArrayType::I64(_flatten(x, axis)?).into()),
        _ => Err(anyhow!("Unsupported input type")),
    }
}
