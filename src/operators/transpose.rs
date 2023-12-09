use crate::{
    common::{BoxResult, OperatorResult, TensorType},
    onnx::NodeProto,
};

const _OPSET_VERSIONS: [i64; 2] = [1, 13];

#[derive(Debug)]
struct TransposeAttrs {
    perm: Vec<usize>,
}

impl TransposeAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            perm: node
                .attribute
                .iter()
                .find(|a| a.name() == "perm")
                .map_or(vec![], |a| a.ints.iter().map(|v| *v as usize).collect()),
        }
    }
}

/// <https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_transpose.py>
/// <https://onnx.ai/onnx/operators/onnx__Transpose.html>
pub fn transpose(
    inputs: &[&TensorType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let data = inputs[0];
    let attrs = TransposeAttrs::new(node);

    match data {
        TensorType::F32(data) => {
            let transposed = if attrs.perm.is_empty() {
                data.t().to_owned()
            } else {
                data.clone().permuted_axes(attrs.perm)
            };
            Ok(TensorType::F32(transposed).into())
        }
        TensorType::I64(data) => {
            let transposed = if attrs.perm.is_empty() {
                data.t().to_owned()
            } else {
                data.clone().permuted_axes(attrs.perm)
            };
            Ok(TensorType::I64(transposed).into())
        }
        _ => todo!("Transpose for type {}", data),
    }
}
