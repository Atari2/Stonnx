use ndarray::ArrayD;

use crate::{
    onnx::NodeProto,
    utils::{ArrayType, BoxResult, NDIndex, OperationResult},
};

const _OPSET_VERSIONS: [i64; 2] = [9, 13];

fn nonzero_generic<A: Default + std::cmp::PartialEq>(input: &ArrayD<A>) -> BoxResult<ArrayD<i64>> {
    let mut result = Vec::<i64>::new();
    for index in NDIndex::new(input.shape()) {
        if input[index.as_slice()] != A::default() {
            result.extend(index.iter().map(|v| *v as i64));
        }
    }
    let internal_dim = input.ndim();
    let result = ArrayD::from_shape_vec(vec![result.len() / internal_dim, internal_dim], result)?;
    Ok(result)
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_non_zero.py
/// https://onnx.ai/onnx/operators/onnx__NonZero.html
pub fn nonzero(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let input = inputs[0];
    match input {
        ArrayType::F32(x) => Ok(ArrayType::I64(nonzero_generic(x)?).into()),
        ArrayType::I64(x) => Ok(ArrayType::I64(nonzero_generic(x)?).into()),
        x => {
            todo!("NonZero for type {}", x);
        }
    }
}
