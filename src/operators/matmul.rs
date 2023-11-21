use ndarray::ArrayD;
use ndarray::Ix2;

use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult, OperationResult};

const _OPSET_VERSIONS: [i64; 3] = [1, 9, 13];

fn matmul_impl(a: ndarray::ArrayViewD<f32>, b: ndarray::ArrayViewD<f32>) -> BoxResult<ArrayD<f32>> {
    if a.ndim() == 2 && b.ndim() == 2 {
        let a = a.into_dimensionality::<Ix2>()?;
        let b = b.into_dimensionality::<Ix2>()?;
        Ok(a.dot(&b).into_dyn())
    } else {
        todo!(
            "Matmul not implemented for ndim {} and {}",
            a.ndim(),
            b.ndim()
        );
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_matmul.py
/// https://onnx.ai/onnx/operators/onnx__MatMul.html
pub fn matmul(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let a = inputs[0];
    let b = inputs[1];

    // FIXME: This needs to be checked at the graph level
    match (a, b) {
        (ArrayType::F32(a), ArrayType::F32(b)) => Ok(ArrayType::F32(matmul_impl(a.view(), b.view())?).into()),
        _ => todo!("Matmul for types {:?} and {:?}", a, b),
    }
}
