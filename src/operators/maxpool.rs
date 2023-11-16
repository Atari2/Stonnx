use ndarray::{s, ArrayD, Ix2, Array2};

use crate::{
    onnx::{AttributeProto, NodeProto},
    utils::{ArrayType, BoxResult},
};

const _OPSET_VERSIONS: [i64; 5] = [1, 8, 10, 11, 12];

#[derive(Debug, PartialEq)]
enum MaxPoolAutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

impl MaxPoolAutoPad {
    fn from_attr(s: &AttributeProto) -> Self {
        match std::str::from_utf8(s.s()) {
            Ok("NOTSET") => Self::NotSet,
            Ok("SAME_UPPER") => Self::SameUpper,
            Ok("SAME_LOWER") => Self::SameLower,
            Ok("VALID") => Self::Valid,
            _ => Self::NotSet,
        }
    }
}

#[derive(Debug)]
struct MaxPoolAttrs {
    auto_pad: MaxPoolAutoPad,
    ceil_mode: bool,
    dilations: Option<Vec<i64>>,
    kernel_shape: Vec<i64>,
    pads: Option<Vec<i64>>,
    storage_order: i64,
    strides: Option<Vec<i64>>,
}
impl MaxPoolAttrs {
    fn new(node: &NodeProto, x: &ArrayType) -> Self {
        Self {
            auto_pad: node
                .attribute
                .iter()
                .find(|a| a.name() == "auto_pad")
                .map_or(MaxPoolAutoPad::NotSet, MaxPoolAutoPad::from_attr),
            ceil_mode: node
                .attribute
                .iter()
                .find(|a| a.name() == "ceil_mode")
                .map_or(false, |a| a.i == Some(1)),
            storage_order: node
                .attribute
                .iter()
                .find(|a| a.name() == "storage_order")
                .map_or(0, |a| a.i.unwrap_or(0)),
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
        }
    }
}

type MaxPoolOutput = (ArrayD<f32>, Option<ArrayD<f32>>);

fn _max_pool_f32_1d(
    input: &ArrayD<f32>,
    attrs: MaxPoolAttrs,
    new_pads: Array2<usize>,
    output_spatial_shape: Vec<usize>,
) -> BoxResult<MaxPoolOutput> {
    todo!()
}

fn _max_pool_f32_2d(
    input: &ArrayD<f32>,
    attrs: MaxPoolAttrs,
    new_pads: Array2<usize>,
    output_spatial_shape: Vec<usize>,
) -> BoxResult<MaxPoolOutput> {
    todo!()
}

fn _max_pool_f32_3d(
    input: &ArrayD<f32>,
    attrs: MaxPoolAttrs,
    new_pads: Array2<usize>,
    output_spatial_shape: Vec<usize>,
) -> BoxResult<MaxPoolOutput> {
    todo!()
}

fn _maxpool_internal_f32(
    input: &ArrayD<f32>,
    attrs: MaxPoolAttrs,
) -> BoxResult<MaxPoolOutput> {
    let pads = if let Some(ref pads) = attrs.pads {
        pads.clone()
    } else {
        vec![0; 2 * attrs.kernel_shape.len()]
    };
    let strides = if let Some(ref strides) = attrs.strides {
        strides.clone()
    } else {
        vec![1; attrs.kernel_shape.len()]
    };
    let dilations = if let Some(ref dilations) = attrs.dilations {
        dilations.clone()
    } else {
        vec![1; attrs.kernel_shape.len()]
    };
    let n_dims = attrs.kernel_shape.len();
    let new_pads_dim = Ix2(n_dims, 2);
    let new_pads_vec = (0..n_dims)
        .flat_map(|i| [pads[i] as usize, pads[i + n_dims] as usize].into_iter())
        .collect::<Vec<_>>();
    let mut new_pads = ndarray::Array2::from_shape_vec(new_pads_dim, new_pads_vec)?;
    let input_spatial_shape = &input.shape()[2..];
    let mut output_spatial_shape = vec![0; input_spatial_shape.len()];
    if attrs.ceil_mode {
        for i in 0..input_spatial_shape.len() {
            output_spatial_shape[i] =
                (((input_spatial_shape[i] as i64 + new_pads.slice(s![i, ..]).sum() as i64
                    - ((attrs.kernel_shape[i] - 1) * dilations[i] + 1)) as f64
                    / strides[i] as f64)
                    + 1f64)
                    .ceil() as usize;
        }
    } else {
        for i in 0..input_spatial_shape.len() {
            output_spatial_shape[i] =
                (((input_spatial_shape[i] as i64 + new_pads.slice(s![i, ..]).sum() as i64
                    - ((attrs.kernel_shape[i] - 1) * dilations[i] + 1)) as f64
                    / strides[i] as f64)
                    + 1f64)
                    .floor() as usize;
        }
    }
    if attrs.auto_pad != MaxPoolAutoPad::NotSet {
        if attrs.auto_pad == MaxPoolAutoPad::SameLower
            || attrs.auto_pad == MaxPoolAutoPad::SameUpper
        {
            for i in 0..input_spatial_shape.len() {
                if attrs.auto_pad == MaxPoolAutoPad::SameUpper {
                    output_spatial_shape[i] =
                        (input_spatial_shape[i] as f64 / strides[i] as f64).ceil() as usize;
                } else {
                    output_spatial_shape[i] =
                        (input_spatial_shape[i] as f64 / strides[i] as f64).floor() as usize;
                }
                let pad_i = (output_spatial_shape[i] - 1) * strides[i] as usize
                    + ((attrs.kernel_shape[i] - 1) * dilations[i] + 1) as usize
                    - input_spatial_shape[i];
                new_pads[[i, 0]] = pad_i / 2;
                new_pads[[i, 1]] = pad_i - new_pads[[i, 0]];
            }
        }
    } else {
        for i in 0..input_spatial_shape.len() {
            output_spatial_shape[i] = ((input_spatial_shape[i] - ((attrs.kernel_shape[i] - 1) * dilations[i] + 1) as usize + 1) as f64 / strides[i] as f64).ceil() as usize;
        }
    }
    match input_spatial_shape.len() {
        1 => _max_pool_f32_1d(input, attrs, new_pads, output_spatial_shape),
        2 => _max_pool_f32_2d(input, attrs, new_pads, output_spatial_shape),
        3 => _max_pool_f32_3d(input, attrs, new_pads, output_spatial_shape),
        _ => todo!("MaxPool for {}D", input_spatial_shape.len()),
    }
}

fn _common_pool_f32(
    input: &ArrayD<f32>,
    attrs: MaxPoolAttrs,
) -> BoxResult<MaxPoolOutput> {
    todo!("Common pool f32")
}

fn maxpool_f32(
    input: &ArrayD<f32>,
    attrs: MaxPoolAttrs,
) -> BoxResult<MaxPoolOutput> {
    let b1 = if let Some(ref dilations) = attrs.dilations {
        let mindilation = dilations.iter().min().copied().unwrap_or(1);
        let maxdilation = dilations.iter().max().copied().unwrap_or(1);
        mindilation != maxdilation || mindilation != 1
    } else {
        false
    };
    let b2 = if let Some(ref strides) = attrs.strides {
        let minstrides = strides.iter().min().copied().unwrap_or(1);
        let maxstrides = strides.iter().max().copied().unwrap_or(1);
        minstrides != maxstrides || minstrides != 1
    } else {
        false
    };
    if b1 || b2 {
        _maxpool_internal_f32(input, attrs)
    } else {
        _common_pool_f32(input, attrs)
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/_op_common_pool.py
/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_max_pool.py
/// https://onnx.ai/onnx/operators/onnx__MaxPool.html
pub fn maxpool(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64,
) -> BoxResult<(ArrayType, Option<ArrayType>)> {
    if inputs.is_empty() {
        return Err("No inputs".into());
    }
    let attrs = MaxPoolAttrs::new(node, inputs[0]);
    match inputs[0] {
        ArrayType::F32(x) => {
            let (y, i) = maxpool_f32(x, attrs)?;
            Ok((ArrayType::F32(y), i.map(ArrayType::F32)))
        }
        _ => todo!("MaxPool for type {}", inputs[0]),
    }
}
