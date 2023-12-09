use ndarray::{ArrayD, ArrayViewD, Ix1};
use num::Zero;

use crate::{
    common::{ArrayType, BoxResult, OperationResult},
    onnx::NodeProto,
    utils::pick_opset_version,
};
use anyhow::anyhow;

const OPSET_VERSIONS: [i64; 3] = [1, 11, 13];

#[derive(Debug)]
struct UnsqueezeAttrs<'a> {
    axes: &'a [i64],
}

impl<'a> UnsqueezeAttrs<'a> {
    fn new(node: &'a NodeProto) -> Self {
        Self {
            axes: node
                .attribute
                .iter()
                .find(|a| a.name() == "axes")
                .map_or(&[], |a| a.ints.as_slice()),
        }
    }
}

fn _unsqueeze_generic<A: Clone + Copy + Zero>(
    data: ArrayViewD<A>,
    new_shape: &[usize],
) -> BoxResult<ArrayD<A>> {
    Ok(data.to_shape(new_shape)?.to_owned())
}

fn unsqueeze_11(inputs: &[&ArrayType], attrs: UnsqueezeAttrs) -> BoxResult<ArrayType> {
    let input = inputs[0];
    let mut shape = input.shape().to_vec();
    for axis in attrs.axes.iter() {
        let axis = if *axis < 0 {
            shape.len() as i64 + axis
        } else {
            *axis
        } as usize;
        shape.insert(axis, 1);
    }
    match input {
        ArrayType::F32(a) => Ok(ArrayType::F32(_unsqueeze_generic(a.view(), &shape)?)),
        ArrayType::I64(a) => Ok(ArrayType::I64(_unsqueeze_generic(a.view(), &shape)?)),
        _ => todo!("Unsqueeze for type {}", input),
    }
}

fn unsqueeze_13(inputs: &[&ArrayType]) -> BoxResult<ArrayType> {
    let input = inputs[0];
    let axes = if let ArrayType::I64(a) = inputs[1] {
        a.clone().into_dimensionality::<Ix1>()?.to_vec()
    } else {
        return Err(anyhow!("Axes must be an int64"));
    };
    let mut shape = input.shape().to_vec();
    for axis in axes.iter() {
        let axis = if *axis < 0 {
            shape.len() as i64 + axis
        } else {
            *axis
        } as usize;
        shape.insert(axis, 1);
    }
    match input {
        ArrayType::F32(a) => Ok(ArrayType::F32(_unsqueeze_generic(a.view(), &shape)?)),
        ArrayType::I64(a) => Ok(ArrayType::I64(_unsqueeze_generic(a.view(), &shape)?)),
        _ => todo!("Unsqueeze for type {}", input),
    }
}

/// <https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_unsqueeze.py>
/// <https://onnx.ai/onnx/operators/onnx__Unsqueeze.html>
pub fn unsqueeze(
    inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    if target_version > 11 {
        Ok(unsqueeze_13(inputs)?.into())
    } else {
        Ok(unsqueeze_11(inputs, UnsqueezeAttrs::new(node))?.into())
    }
}
