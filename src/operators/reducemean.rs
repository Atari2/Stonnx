use ndarray::Axis;

use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult, OperationResult, pick_opset_version};

const OPSET_VERSIONS: [i64; 4] = [1, 11, 13, 18];

struct ReduceMeanAttrs {
    axes: Vec<i64>,
    keepdims: i64,
    noop_with_empty_axes: i64,
}

impl ReduceMeanAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            axes: node
                .attribute
                .iter()
                .find(|a| a.name() == "axes")
                .map_or(vec![], |a| a.ints.clone()),
            keepdims: node
                .attribute
                .iter()
                .find(|a| a.name() == "keepdims")
                .map_or(1, |a| a.i.unwrap_or(1)),
            noop_with_empty_axes: node
                .attribute
                .iter()
                .find(|a| a.name() == "noop_with_empty_axes")
                .map_or(0, |a| a.i.unwrap_or(0)),
        }
    }
}

fn reducemean_1(inputs: &[&ArrayType], attrs: ReduceMeanAttrs) -> BoxResult<ArrayType> {
    let axes = attrs.axes;
    let a = inputs[0];
    let keepdims = attrs.keepdims;

    // if axis is < 0, count from the back
    let axes: Vec<i64> = axes.iter().map(|&axis| if axis < 0 { a.ndim() as i64 + axis } else { axis }).collect();

    match a {
        ArrayType::F32(a) => {
            let mut a = a.to_owned();

            let mut reduced_dims: Vec<usize> = Vec::new();

            for axis in axes {
                if let Some(mean_axis) = a.mean_axis(Axis(axis as usize)) {
                    a = mean_axis;
                    reduced_dims.push(axis as usize);
                } else {
                    return Err("Error computing mean along axis".into());
                }
            }
            let new_shape: Vec<usize> = a.shape().iter().filter(|&&dim| dim > 1).cloned().collect();

            if keepdims == 0 {
                a = a.into_shape(new_shape.clone()).expect("Failed to reshape array");
            }
            else {
                // Add back the dimensions that were reduced
                for &dim in &reduced_dims {
                    a = a.insert_axis(Axis(dim));
                }
            }

            Ok(ArrayType::F32(a))
        }
        _ => todo!("ReduceMean for type {:?}", a),
    }
}


// FIX ME: This is a copy of reducemean_1, but with noop_with_empty_axes and axes as inputs
fn reducemean_18(inputs: &[&ArrayType], attrs: ReduceMeanAttrs) -> BoxResult<ArrayType> {
    let a = inputs[0];
    let keepdims = attrs.keepdims;
    let noop_with_empty_axes = attrs.noop_with_empty_axes;

    let axes = if noop_with_empty_axes == 1 {
        attrs.axes
    } else {
        let mut axes = attrs.axes;
        if axes.is_empty() {
            axes = (0..a.ndim() as i64).collect();
        }
        axes
    };

    let axes: Vec<i64> = axes.iter().map(|&axis| if axis < 0 { a.ndim() as i64 + axis } else { axis }).collect();

    match a {
        ArrayType::F32(a) => {
            let mut a = a.to_owned();
            let mut reduced_dims: Vec<usize> = Vec::new();

            for axis in axes {
                if let Some(mean_axis) = a.mean_axis(Axis(axis as usize)) {
                    a = mean_axis;
                    reduced_dims.push(axis as usize);
                } else {
                    return Err("Error computing mean along axis".into());
                }
            }
            let new_shape: Vec<usize> = a.shape().iter().filter(|&&dim| dim > 1).cloned().collect();

            if keepdims == 0 {
                a = a.into_shape(new_shape.clone()).expect("Failed to reshape array");
            }
            else {
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
