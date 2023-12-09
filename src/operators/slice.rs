use itertools::izip;
use ndarray::{Array1, ArrayD, Axis, SliceInfoElem};

use crate::common::{ArrayType, BoxResult, OperationResult};
use crate::onnx::NodeProto;
use crate::utils::pick_opset_version;
use anyhow::anyhow;

const OPSET_VERSIONS: [i64; 4] = [1, 10, 11, 13];

struct SliceAttrs<'a> {
    axes: Option<&'a [i64]>,
    ends: &'a [i64],
    starts: &'a [i64],
}

impl<'a> SliceAttrs<'a> {
    fn new(node: &'a NodeProto) -> Self {
        Self {
            axes: node
                .attribute
                .iter()
                .find(|a| a.name() == "axes")
                .map(|a| a.ints.as_slice()),
            ends: node
                .attribute
                .iter()
                .find(|a| a.name() == "ends")
                .map_or(&[], |a| a.ints.as_slice()),
            starts: node
                .attribute
                .iter()
                .find(|a| a.name() == "starts")
                .map_or(&[], |a| a.ints.as_slice()),
        }
    }
}

fn common_slice_f32(
    data: &ArrayD<f32>,
    axes: Option<&ArrayD<i64>>,
    starts: &ArrayD<i64>,
    ends: &ArrayD<i64>,
    steps: Option<&ArrayD<i64>>,
) -> BoxResult<ArrayD<f32>> {
    let starts = if starts.shape().is_empty() {
        starts.clone().insert_axis(Axis(0))
    } else {
        starts.clone()
    };
    let ends = if ends.shape().is_empty() {
        ends.clone().insert_axis(Axis(0))
    } else {
        ends.clone()
    };

    let slices = if let Some(axes) = axes {
        let mut slices = data
            .shape()
            .iter()
            .map(|a| (0..*a).into())
            .collect::<Vec<SliceInfoElem>>();
        if let Some(steps) = steps {
            for (&s, &e, &a, &d) in izip!(starts.iter(), ends.iter(), axes.iter(), steps.iter()) {
                let a = if a < 0 { a + data.ndim() as i64 } else { a } as usize;
                if e > data.shape()[a] as i64 {
                    slices[a] = SliceInfoElem::Slice {
                        start: s as isize,
                        end: None,
                        step: d as isize,
                    };
                } else {
                    slices[a] = SliceInfoElem::Slice {
                        start: s as isize,
                        end: Some(e as isize),
                        step: d as isize,
                    };
                }
            }
        } else {
            for (&s, &e, &a) in izip!(starts.iter(), ends.iter(), axes.iter()) {
                let a = if a < 0 { a + data.ndim() as i64 } else { a } as usize;
                if e > data.shape()[a] as i64 {
                    slices[a] = SliceInfoElem::Slice {
                        start: s as isize,
                        end: None,
                        step: 1,
                    };
                } else {
                    slices[a] = SliceInfoElem::Slice {
                        start: s as isize,
                        end: Some(e as isize),
                        step: 1,
                    };
                }
            }
        }
        slices
    } else if let Some(steps) = steps {
        izip!(starts.iter(), ends.iter(), steps.iter())
            .enumerate()
            .map(|(i, (&s, &e, &d))| {
                let ee = if e > data.shape()[i] as i64 {
                    None
                } else {
                    Some(e as isize)
                };
                SliceInfoElem::Slice {
                    start: s as isize,
                    end: ee,
                    step: d as isize,
                }
            })
            .collect::<Vec<_>>()
    } else {
        izip!(starts.iter(), ends.iter())
            .enumerate()
            .map(|(i, (&s, &e))| {
                let ee = if e > data.shape()[i] as i64 {
                    None
                } else {
                    Some(e as isize)
                };
                SliceInfoElem::Slice {
                    start: s as isize,
                    end: ee,
                    step: 1,
                }
            })
            .collect::<Vec<SliceInfoElem>>()
    };
    Ok(data.slice(slices.as_slice()).to_owned())
}

fn slice_1(inputs: &[&ArrayType], attrs: SliceAttrs) -> BoxResult<ArrayType> {
    let data = if let ArrayType::F32(data) = inputs[0] {
        data
    } else {
        return Err(anyhow!("Slice1 only support f32"));
    };
    let axes = attrs
        .axes
        .map(|axes| Array1::<i64>::from_vec(axes.to_vec()).into_dyn());
    let starts = Array1::<i64>::from_vec(attrs.starts.to_vec()).into_dyn();
    let ends = Array1::<i64>::from_vec(attrs.ends.to_vec()).into_dyn();
    Ok(ArrayType::F32(common_slice_f32(
        data,
        axes.as_ref(),
        &starts,
        &ends,
        None,
    )?))
}

fn slice_10(inputs: &[&ArrayType]) -> BoxResult<ArrayType> {
    let data = if let ArrayType::F32(data) = inputs[0] {
        data
    } else {
        return Err(anyhow!("Slice1 only support f32"));
    };
    let starts = if let ArrayType::I64(starts) = inputs[1] {
        starts
    } else {
        return Err(anyhow!("Slice1 only support i64"));
    };
    let ends = if let ArrayType::I64(ends) = inputs[2] {
        ends
    } else {
        return Err(anyhow!("Slice1 only support i64"));
    };
    let axes = match inputs.get(3) {
        Some(ArrayType::I64(data)) => Some(data),
        None => None,
        _ => return Err(anyhow!("Slice1 only support i64")),
    };
    let steps = match inputs.get(4) {
        Some(ArrayType::I64(steps)) => Some(steps),
        None => None,
        _ => return Err(anyhow!("Slice1 only support i64")),
    };
    Ok(ArrayType::F32(common_slice_f32(
        data, axes, starts, ends, steps,
    )?))
}

/// <https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_slice.py>
/// <https://onnx.ai/onnx/operators/onnx__Slice.html>
pub fn slice(
    inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let target_ver = pick_opset_version(opset_version, &OPSET_VERSIONS);
    if target_ver < 10 {
        let attrs = SliceAttrs::new(node);
        Ok(slice_1(inputs, attrs)?.into())
    } else {
        Ok(slice_10(inputs)?.into())
    }
}
