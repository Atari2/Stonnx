use ndarray::{ArrayD, IxDyn};

use crate::{
    common::{BoxResult, OperatorResult, TensorType},
    onnx::NodeProto,
    utils::make_tensor_from_proto,
};
use anyhow::anyhow;

const _OPSET_VERSIONS: [i64; 2] = [9, 20];

#[derive(Debug)]
struct ConstantOfShapeAttrs {
    value: TensorType,
}

impl ConstantOfShapeAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            value: node
                .attribute
                .iter()
                .find(|a| a.name() == "value")
                .map_or_else(
                    || TensorType::F32(ArrayD::zeros(IxDyn(&[]))),
                    |a| {
                        make_tensor_from_proto(&a.t)
                            .unwrap_or_else(|_| TensorType::F32(ArrayD::zeros(IxDyn(&[]))))
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

/// Generate a tensor with given value and shape.
///
/// [Python reference](<https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_constant_of_shape.py>)
///
/// [ONNX Documentation](<https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html>)
pub fn constantofshape(
    input: &[&TensorType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let shape = if let Some(TensorType::I64(shape)) = input.get(0) {
        shape.iter().map(|v| *v as usize).collect::<Vec<_>>()
    } else {
        return Err(anyhow!("ConstantOfShape: shape must be i64"));
    };
    let attrs = ConstantOfShapeAttrs::new(node);
    match attrs.value {
        TensorType::F32(v) => Ok(TensorType::F32(_constanofshape_generic(&shape, v)?).into()),
        TensorType::I64(v) => Ok(TensorType::I64(_constanofshape_generic(&shape, v)?).into()),
        _ => Err(anyhow!(
            "ConstantOfShape: value must be f32, i64, u8 or bool"
        )),
    }
}
