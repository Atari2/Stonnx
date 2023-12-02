use ndarray::Axis;

use crate::common::{ArrayType, BoxResult, OperationResult};
use crate::onnx::NodeProto;
use crate::utils::pick_opset_version;
use anyhow::anyhow;

const OPSET_VERSIONS: [i64; 4] = [1, 11, 13, 18];

struct ReduceMeanAttrs<'a> {
    axes: Option<&'a [i64]>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

impl<'a> ReduceMeanAttrs<'a> {
    fn new(node: &'a NodeProto) -> Self {
        Self {
            axes: node
                .attribute
                .iter()
                .find(|a| a.name() == "axes")
                .map(|a| a.ints.as_slice()),
            keepdims: node
                .attribute
                .iter()
                .find(|a| a.name() == "keepdims")
                .map_or(true, |a| a.i.unwrap_or(1) == 1),
            noop_with_empty_axes: node
                .attribute
                .iter()
                .find(|a| a.name() == "noop_with_empty_axes")
                .map_or(false, |a| a.i.unwrap_or(0) == 1),
        }
    }
}

fn reducemean_1(inputs: &[&ArrayType], attrs: ReduceMeanAttrs) -> BoxResult<ArrayType> {
    let a = inputs[0];

    match a {
        ArrayType::F32(a) => {
            let mut a = a.to_owned();

            let mut reduced_dims: Vec<usize> = Vec::new();
            if let Some(axes) = attrs.axes {
                for axis in axes {
                    let axis = if *axis < 0 {
                        a.ndim() as i64 + axis
                    } else {
                        *axis
                    } as usize;
                    if let Some(mean_axis) = a.mean_axis(Axis(axis)) {
                        a = mean_axis;
                        reduced_dims.push(axis);
                    } else {
                        return Err(anyhow!("Error computing mean along axis"));
                    }
                }
            }
            let new_shape: Vec<usize> = a.shape().iter().filter(|&&dim| dim > 1).cloned().collect();

            if !attrs.keepdims {
                a = a.into_shape(new_shape.clone())?;
            } else {
                // Add back the dimensions that were reduced
                for dim in reduced_dims.iter() {
                    a = a.insert_axis(Axis(*dim));
                }
            }

            Ok(ArrayType::F32(a))
        }
        _ => todo!("ReduceMean for type {:?}", a),
    }
}

fn reducemean_18(inputs: &[&ArrayType], attrs: ReduceMeanAttrs) -> BoxResult<ArrayType> {
    let a = inputs[0];

    if attrs.axes.is_none() && attrs.noop_with_empty_axes {
        // no-op
        return Ok(a.to_owned());
    }

    let axes = attrs.axes.filter(|&axes| !axes.is_empty());

    match a {
        ArrayType::F32(a) => {
            let mut a = a.to_owned();
            let mut reduced_dims: Vec<usize> = Vec::new();
            if let Some(axes) = axes {
                for axis in axes {
                    let axis = if *axis < 0 {
                        a.ndim() as i64 + axis
                    } else {
                        *axis
                    } as usize;
                    if let Some(mean_axis) = a.mean_axis(Axis(axis)) {
                        a = mean_axis;
                        reduced_dims.push(axis);
                    } else {
                        return Err(anyhow!("Error computing mean along axis"));
                    }
                }
            }
            let new_shape: Vec<usize> = a.shape().iter().filter(|&&dim| dim > 1).cloned().collect();

            if !attrs.keepdims {
                a = a.into_shape(new_shape.clone())?;
            } else {
                for &dim in &reduced_dims {
                    a = a.insert_axis(Axis(dim));
                }
            }

            Ok(ArrayType::F32(a))
        }
        _ => todo!("ReduceMean for type {:?}", a),
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_reduce_mean.py
/// https://onnx.ai/onnx/operators/onnx__ReduceMean.html
pub fn reducemean(
    inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    if target_version < 18 {
        Ok(reducemean_1(inputs, ReduceMeanAttrs::new(node))?.into())
    } else {
        Ok(reducemean_18(inputs, ReduceMeanAttrs::new(node))?.into())
    }
}
