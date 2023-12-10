use ndarray::ArrayD;

use crate::{
    common::{BoxResult, NDIndex, OperatorResult, TensorType},
    onnx::NodeProto,
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
    let result = ArrayD::from_shape_vec(vec![internal_dim, result.len() / internal_dim], result)?;
    Ok(result)
}

/// Returns the indices of the elements that are non-zero (in row-major order - by dimension).
///
/// NonZero behaves similar to numpy.nonzero: <https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html>, but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is different from Numpy’s behavior.
///
/// [Python reference](<https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_non_zero.py>)
///
/// [ONNX Documentation](<https://onnx.ai/onnx/operators/onnx__NonZero.html>)
pub fn nonzero(
    inputs: &[&TensorType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let input = inputs[0];
    match input {
        TensorType::F32(x) => Ok(TensorType::I64(nonzero_generic(x)?).into()),
        TensorType::I64(x) => Ok(TensorType::I64(nonzero_generic(x)?).into()),
        x => {
            todo!("NonZero for type {}", x);
        }
    }
}
