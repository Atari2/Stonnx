use ndarray::{Dimension, ArrayD, Array, IxDyn};

use crate::{
    onnx::NodeProto,
    utils::{ArrayType, BoxResult, OperationResult},
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

fn transpose_<T, D>(data: Array<T, D>, perm: Option<Vec<usize>>) -> ArrayD<T>
where
    T: Clone,
    D: Dimension,
{
    match perm {
        Some(perm) => todo!(),
        None => todo!(),
    }
}



/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_transpose.py
/// https://onnx.ai/onnx/operators/onnx__Transpose.html
pub fn transpose(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let data = inputs[0];
    let attrs = TransposeAttrs::new(node);

    let attrs = TransposeAttrs::new(node);
    let perm = if attrs.perm.is_empty() { None } else { Some(attrs.perm) };
    

    todo!()

}
