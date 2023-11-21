use crate::{
    onnx::NodeProto,
    utils::{ArrayType, BoxResult, OperationResult},
};
use anyhow::anyhow;

use super::_commonpool::{CommonPoolAttrs, PoolAutoPad, PoolingType, _common_pool_f32};

const _OPSET_VERSIONS: [i64; 5] = [1, 7, 10, 11, 19];

#[derive(Debug)]
struct AveragePoolAttrs {
    auto_pad: Option<PoolAutoPad>,
    ceil_mode: bool,
    count_include_pad: bool,
    dilations: Option<Vec<i64>>,
    kernel_shape: Vec<i64>,
    pads: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
}
impl AveragePoolAttrs {
    fn new(node: &NodeProto, x: &ArrayType) -> Self {
        Self {
            auto_pad: node
                .attribute
                .iter()
                .find(|a| a.name() == "auto_pad")
                .map(PoolAutoPad::from_attr),
            ceil_mode: node
                .attribute
                .iter()
                .find(|a| a.name() == "ceil_mode")
                .map_or(false, |a| a.i == Some(1)),
            dilations: node
                .attribute
                .iter()
                .find(|a| a.name() == "dilations")
                .map(|a| a.ints.to_vec()),
            kernel_shape: node
                .attribute
                .iter()
                .find(|a| a.name() == "kernel_shape")
                .map_or_else(
                    || x.shape().iter().skip(2).map(|x| *x as i64).collect(),
                    |a| a.ints.to_vec(),
                ),
            pads: node
                .attribute
                .iter()
                .find(|a| a.name() == "pads")
                .map(|a| a.ints.to_vec()),
            strides: node
                .attribute
                .iter()
                .find(|a| a.name() == "strides")
                .map(|a| a.ints.to_vec()),
            count_include_pad: node
                .attribute
                .iter()
                .find(|a| a.name() == "count_include_pad")
                .map_or(false, |a| a.i == Some(1)),
        }
    }
}

impl From<AveragePoolAttrs> for CommonPoolAttrs {
    fn from(attrs: AveragePoolAttrs) -> Self {
        Self {
            auto_pad: attrs.auto_pad,
            ceil_mode: attrs.ceil_mode,
            dilations: attrs.dilations,
            kernel_shape: attrs.kernel_shape,
            pads: attrs.pads,
            storage_order: 0,
            strides: attrs.strides,
        }
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_average_pool.py
/// https://onnx.ai/onnx/operators/onnx__AveragePool.html
pub fn averagepool(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64,
    output_len: usize,
) -> BoxResult<OperationResult> {
    if inputs.is_empty() {
        return Err(anyhow!("No inputs"));
    }
    let attrs = AveragePoolAttrs::new(node, inputs[0]);
    match inputs[0] {
        ArrayType::F32(x) => {
            let (y, i) = _common_pool_f32(
                x,
                PoolingType::Average,
                attrs.count_include_pad as i64,
                attrs.into(),
                output_len,
            )?;
            Ok((ArrayType::F32(y), i.map(ArrayType::I64)).into())
        }
        _ => todo!("AveragePool for type {:?}", inputs[0]),
    }
}
