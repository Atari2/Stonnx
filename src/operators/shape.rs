use crate::common::{BoxResult, OperatorResult, TensorType};
use crate::onnx::NodeProto;
use crate::utils::pick_opset_version;

const OPSET_VERSIONS: [i64; 4] = [1, 13, 15, 19];

#[derive(Debug)]
struct ShapeAttrs {
    start: i64,
    end: Option<i64>,
}

impl ShapeAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            start: node
                .attribute
                .iter()
                .find(|a| a.name() == "start")
                .map_or(0, |a| a.i.unwrap_or(0)),
            end: node
                .attribute
                .iter()
                .find(|a| a.name() == "end")
                .and_then(|a| a.i),
        }
    }
}

fn shape_1(inputs: &[&TensorType]) -> BoxResult<TensorType> {
    let input = inputs[0];
    let shape = input.shape();
    let output_shape = ndarray::IxDyn(&[shape.len()]);
    Ok(TensorType::I64(ndarray::ArrayD::from_shape_vec(
        output_shape,
        shape.iter().map(|v| *v as i64).collect(),
    )?))
}

fn interval(n: i64, start: i64, end: Option<i64>) -> Option<(i64, i64)> {
    if start == 0 {
        if let Some(e) = end {
            if e < 0 {
                Some((0, n + e))
            } else {
                Some((0, e))
            }
        } else {
            None
        }
    } else if let Some(e) = end {
        if e < 0 {
            Some((start, n + e))
        } else {
            Some((start, e))
        }
    } else {
        Some((start, n))
    }
}

fn shape_15(inputs: &[&TensorType], attrs: ShapeAttrs) -> BoxResult<TensorType> {
    let data = inputs[0];
    let shape = data.shape();
    let absome = interval(data.shape().len() as i64, attrs.start, attrs.end);
    if let Some((a, b)) = absome {
        let shape = &shape[a as usize..b as usize];
        let output_shape = ndarray::IxDyn(&[shape.len()]);
        Ok(TensorType::I64(ndarray::ArrayD::from_shape_vec(
            output_shape,
            shape.iter().map(|v| *v as i64).collect(),
        )?))
    } else {
        shape_1(inputs)
    }
}

/// <https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_shape.py>
/// <https://onnx.ai/onnx/operators/onnx__Shape.html>
pub fn shape(
    inputs: &[&TensorType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    if target_version < 15 {
        Ok(shape_1(inputs)?.into())
    } else {
        Ok(shape_15(inputs, ShapeAttrs::new(node))?.into())
    }
}
