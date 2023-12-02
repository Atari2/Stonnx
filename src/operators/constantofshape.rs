use ndarray::{ArrayD, IxDyn};

use crate::{
    common::{ArrayType, BoxResult, OperationResult},
    onnx::NodeProto,
    utils::make_tensor_from_proto,
};
use anyhow::anyhow;

const _OPSET_VERSIONS: [i64; 2] = [9, 20];

#[derive(Debug)]
struct ConstantOfShapeAttrs {
    value: ArrayType,
}

impl ConstantOfShapeAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            value: node
                .attribute
                .iter()
                .find(|a| a.name() == "value")
                .map_or_else(
                    || ArrayType::F32(ArrayD::zeros(IxDyn(&[]))),
                    |a| {
                        make_tensor_from_proto(&a.t)
                            .unwrap_or_else(|_| ArrayType::F32(ArrayD::zeros(IxDyn(&[]))))
                    },
                ),
        }
    }
}

fn _constanofshape_generic<A: Clone + std::iter::Sum<A> + Copy>(
    shape: &[usize],
    value: ArrayD<A>,
) -> BoxResult<ArrayD<A>> {
    let scalar_value = value.iter().copied().sum();
    Ok(ArrayD::from_elem(IxDyn(shape), scalar_value))
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_constant_of_shape.py
/// https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html
pub fn constantofshape(
    input: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let shape = if let Some(ArrayType::I64(shape)) = input.get(0) {
        shape.iter().map(|v| *v as usize).collect::<Vec<_>>()
    } else {
        return Err(anyhow!("ConstantOfShape: shape must be i64"));
    };
    let attrs = ConstantOfShapeAttrs::new(node);
    match attrs.value {
        ArrayType::F32(v) => Ok(ArrayType::F32(_constanofshape_generic(&shape, v)?).into()),
        ArrayType::I64(v) => Ok(ArrayType::I64(_constanofshape_generic(&shape, v)?).into()),
        _ => Err(anyhow!(
            "ConstantOfShape: value must be f32, i64, u8 or bool"
        )),
    }
}
