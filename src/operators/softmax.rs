use ndarray::{ArrayD, Axis};

use crate::{
    common::OperatorResult,
    common::{BoxResult, TensorType},
    onnx::NodeProto,
};

const _OPSET_VERSIONS: [i64; 3] = [1, 11, 13];

#[derive(Debug)]
struct SoftmaxAttrs {
    axis: i64,
}

impl SoftmaxAttrs {
    fn new(node: &NodeProto, version: i64) -> Self {
        Self {
            axis: node
                .attribute
                .iter()
                .find(|a| a.name() == "axis")
                .map_or_else(
                    || {
                        if version < 13 {
                            1
                        } else {
                            -1
                        }
                    },
                    |a| a.i.unwrap_or(if version < 13 { 1 } else { -1 }),
                ),
        }
    }
}

fn _softmax_f32(input: &ArrayD<f32>, axis: usize) -> BoxResult<TensorType> {
    let new_shape = input
        .shape()
        .iter()
        .enumerate()
        .map(|(i, &s)| if i == axis { 1 } else { s })
        .collect::<Vec<_>>();
    let tmpmax = input
        .map_axis(Axis(axis), |a| a.iter().fold(f32::MIN, |a, &b| a.max(b)))
        .into_shape(new_shape)?;
    let mut y = input - &tmpmax;
    y.mapv_inplace(|a| a.exp());
    let ynewshape = y
        .shape()
        .iter()
        .enumerate()
        .map(|(i, &s)| if i == axis { 1 } else { s })
        .collect::<Vec<_>>();
    let ysum = y.sum_axis(Axis(axis)).into_shape(ynewshape)?.to_owned();
    y = y / ysum;
    Ok(TensorType::F32(y))
}

/// The operator computes the normalized exponential values for the given input:
/// Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
///
/// The “axis” attribute indicates the dimension along which Softmax will be performed.
/// The output tensor has the same shape and contains the Softmax values of the corresponding input.
///
/// [Python reference](<https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_softmax.py>)
///
/// [ONNX Documentation](<https://onnx.ai/onnx/operators/onnx__Softmax.html>)
pub fn softmax(
    inputs: &[&TensorType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let attrs = SoftmaxAttrs::new(node, opset_version);
    let data = inputs[0];

    let axis = if attrs.axis < 0 {
        data.shape().len() as i64 + attrs.axis
    } else {
        attrs.axis
    } as usize;

    match data {
        TensorType::F32(data) => Ok(_softmax_f32(data, axis)?.into()),
        _ => todo!("Softmax for type {}", data),
    }
}
