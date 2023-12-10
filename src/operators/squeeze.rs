use ndarray::{ArrayD, ArrayViewD, Ix1};
use num::Zero;

use crate::{
    common::{BoxResult, OperatorResult, TensorType},
    onnx::NodeProto,
    utils::pick_opset_version,
};

const OPSET_VERSIONS: [i64; 3] = [1, 11, 13];

#[derive(Debug)]
struct SqueezeAttrs<'a> {
    axes: Option<&'a [i64]>,
}

impl<'a> SqueezeAttrs<'a> {
    fn new(node: &'a NodeProto) -> Self {
        Self {
            axes: node
                .attribute
                .iter()
                .find(|a| a.name() == "axes")
                .and_then(|a| {
                    if a.ints.is_empty() {
                        None
                    } else {
                        Some(a.ints.as_slice())
                    }
                }),
        }
    }
}

fn _squeeze_generic<A: Clone + Copy + Zero>(
    data: ArrayViewD<A>,
    new_shape: &[usize],
) -> BoxResult<ArrayD<A>> {
    Ok(data.to_shape(new_shape).unwrap().to_owned())
}

fn squeeze_11(inputs: &[&TensorType], attrs: SqueezeAttrs) -> BoxResult<TensorType> {
    let input = inputs[0];
    let input_shape = input.shape();
    let shape = if let Some(axes) = attrs.axes {
        let mut shape = input_shape.to_vec();
        for axis in axes.iter() {
            let axis = if *axis < 0 {
                *axis + shape.len() as i64
            } else {
                *axis
            } as usize;
            shape.remove(axis);
        }
        shape
    } else {
        input_shape.iter().filter(|&x| *x != 1).cloned().collect()
    };

    match input {
        TensorType::F32(a) => Ok(TensorType::F32(_squeeze_generic(a.view(), &shape).unwrap())),
        TensorType::I64(a) => Ok(TensorType::I64(_squeeze_generic(a.view(), &shape).unwrap())),
        _ => todo!("Squeeze for type {}", input),
    }
}

fn squeeze_13(inputs: &[&TensorType]) -> BoxResult<TensorType> {
    let input = inputs[0];
    let axes = if let Some(TensorType::I64(a)) = inputs.get(1) {
        a.clone().into_dimensionality::<Ix1>()?.to_vec()
    } else {
        input
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v == 1 { Some(i as i64) } else { None })
            .collect::<Vec<_>>()
    };
    let mut shape = input.shape().to_vec();
    for axis in axes.iter() {
        let axis = if *axis < 0 {
            *axis + shape.len() as i64
        } else {
            *axis
        } as usize;
        shape.remove(axis);
    }
    match input {
        TensorType::F32(a) => Ok(TensorType::F32(_squeeze_generic(a.view(), &shape).unwrap())),
        TensorType::I64(a) => Ok(TensorType::I64(_squeeze_generic(a.view(), &shape).unwrap())),
        _ => todo!("Squeeze for type {}", input),
    }
}

/// Remove single-dimensional entries from the shape of a tensor.
///
/// Takes an input axes with a list of axes to squeeze.
/// If axes is not provided, all the single dimensions will be removed from the shape.
/// If an axis is selected with shape entry not equal to one, an error is raised.
///
/// [Python reference](<https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_squeeze.py>)
///
/// [ONNX Documentation](<https://onnx.ai/onnx/operators/onnx__Squeeze.html>)
pub fn squeeze(
    inputs: &[&TensorType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    if target_version > 11 {
        Ok(squeeze_13(inputs).unwrap().into())
    } else {
        Ok(squeeze_11(inputs, SqueezeAttrs::new(node)).unwrap().into())
    }
}
