#![allow(unused_variables)]
use ndarray::{s, Array2, ArrayD, Ix1, Ix2, IxDyn};

use crate::{
    onnx::NodeProto,
    utils::{shape_safe_product, ArrayType, BoxResult, OperationResult},
};
use anyhow::anyhow;

use super::_commonpool::{CommonPoolAttrs, PoolAutoPad, PoolOutput, PoolingType, _common_pool_f32};

const _OPSET_VERSIONS: [i64; 5] = [1, 8, 10, 11, 12];

#[derive(Debug)]
struct MaxPoolAttrs {
    auto_pad: Option<PoolAutoPad>,
    ceil_mode: bool,
    dilations: Option<Vec<i64>>,
    kernel_shape: Vec<i64>,
    pads: Option<Vec<i64>>,
    storage_order: i64,
    strides: Option<Vec<i64>>,
}
impl MaxPoolAttrs {
    fn replace_dilations_and_strides(&mut self, dilations: Vec<i64>, strides: Vec<i64>) {
        self.dilations = Some(dilations);
        self.strides = Some(strides);
    }
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

impl From<MaxPoolAttrs> for CommonPoolAttrs {
    fn from(attrs: MaxPoolAttrs) -> Self {
        Self {
            auto_pad: attrs.auto_pad,
            ceil_mode: attrs.ceil_mode,
            dilations: attrs.dilations,
            kernel_shape: attrs.kernel_shape,
            pads: attrs.pads,
            storage_order: attrs.storage_order,
            strides: attrs.strides,
        }
    }
}

fn _max_pool_f32_1d(
    input: &ArrayD<f32>,
    attrs: MaxPoolAttrs,
    new_pads: Array2<usize>,
    output_spatial_shape: Vec<usize>,
    output_len: usize,
) -> BoxResult<PoolOutput> {
    todo!()
}

fn _max_pool_f32_2d(
    input: &ArrayD<f32>,
    attrs: MaxPoolAttrs,
    new_pads: Array2<usize>,
    mut output_spatial_shape: Vec<usize>,
    output_len: usize,
) -> BoxResult<PoolOutput> {
    let global_pooling = false;
    let mut y_dims = input.shape()[..2].to_vec();
    y_dims.append(&mut output_spatial_shape);
    let y = ArrayD::<f32>::zeros(IxDyn(&y_dims));
    let indices = ArrayD::<i64>::from_shape_simple_fn(IxDyn(&y_dims), || -1);
    let x_dims = input.shape();
    let channels = x_dims[1];
    let height = x_dims[2];
    let width = if attrs.kernel_shape.len() > 1 {
        x_dims[3]
    } else {
        1
    };
    let pooled_height = y_dims[2];
    let pooled_width = if attrs.kernel_shape.len() > 1 {
        y_dims[3]
    } else {
        1
    };
    let total_channels = x_dims[0] * channels;
    let stride_h = if global_pooling {
        1
    } else if let Some(ref strides) = attrs.strides {
        strides[0]
    } else {
        return Err(anyhow!("Stride not set"));
    };
    let stride_w = if global_pooling {
        1
    } else if let Some(ref strides) = attrs.strides {
        strides[1]
    } else {
        return Err(anyhow!("Stride not set"));
    };
    let x_step = height * width;
    let y_step = pooled_height * pooled_width;
    let dilation_h = if let Some(ref dilations) = attrs.dilations {
        dilations[0]
    } else {
        return Err(anyhow!("Dilations not set"));
    };
    let dilation_w = if let Some(ref dilations) = attrs.dilations {
        dilations[1]
    } else {
        return Err(anyhow!("Dilations not set"));
    };
    let x_data = input.to_shape(Ix1(shape_safe_product(input.shape())))?;
    let mut y_data = y.to_shape(Ix1(shape_safe_product(y.shape())))?;
    let mut i_data = indices.to_shape(Ix1(shape_safe_product(indices.shape())))?;

    let mut iteration = |c: usize| {
        let x_d = c * x_step;
        let y_d = c * y_step;
        for ph in 0..pooled_height {
            let hstart = ph * stride_h as usize - new_pads[[0, 0]];
            let hend = hstart + (attrs.kernel_shape[0] * dilation_h) as usize;
            for pw in 0..pooled_width {
                let wstart = pw * stride_w as usize - new_pads[[1, 0]];
                let wend = wstart + (attrs.kernel_shape[1] * dilation_w) as usize;
                let pool_index = ph * pooled_width + pw;
                let mut y_h = None;
                let mut h_index = -1;
                let mut w_index = -1;
                for h in (hstart..hend).step_by(dilation_h as usize) {
                    if h >= height {
                        continue;
                    }
                    for w in (wstart..wend).step_by(dilation_w as usize) {
                        if w >= width {
                            continue;
                        }
                        let input_index = h * width + w;
                        if input_index > x_data.shape()[0] {
                            continue;
                        }
                        if y_h.is_none() {
                            let y_hv = x_data[x_d + input_index];
                            y_h = Some(y_hv);
                            h_index = h as i64;
                            w_index = w as i64;
                        } else if let Some(ref mut y_h) = y_h {
                            if x_data[x_d + input_index] > *y_h {
                                *y_h = x_data[x_d + input_index];
                                h_index = h as i64;
                                w_index = w as i64;
                            }
                        }
                    }
                }
                if y_h.is_none() {
                    continue;
                } else if let Some(ref mut y_h) = y_h {
                    y_data[y_d + pool_index] = *y_h;
                    i_data[y_d + pool_index] = if attrs.storage_order == 0 {
                        c * x_step + h_index as usize * width + w_index as usize
                    } else {
                        c * x_step + h_index as usize + w_index as usize * height
                    } as i64;
                }
            }
        }
    };

    for c in 0..total_channels {
        iteration(c);
    }
    if output_len == 1 {
        Ok((y_data.to_shape(y_dims)?.to_owned(), None))
    } else {
        Ok((
            y_data.to_shape(y_dims.to_vec())?.to_owned(),
            Some(indices.to_shape(y_dims)?.to_owned()),
        ))
    }
}

fn _max_pool_f32_3d(
    input: &ArrayD<f32>,
    attrs: MaxPoolAttrs,
    new_pads: Array2<usize>,
    output_spatial_shape: Vec<usize>,
    output_len: usize,
) -> BoxResult<PoolOutput> {
    todo!()
}

fn _maxpool_internal_f32(
    input: &ArrayD<f32>,
    mut attrs: MaxPoolAttrs,
    output_len: usize,
) -> BoxResult<PoolOutput> {
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
            let ossv = ((input_spatial_shape[i] + new_pads.slice(s![i, ..]).sum()
                - ((attrs.kernel_shape[i] - 1) * dilations[i] + 1) as usize)
                as f64
                / strides[i] as f64
                + 1f64)
                .floor() as usize;
            output_spatial_shape[i] = ossv;
        }
    }

    match attrs.auto_pad {
        Some(PoolAutoPad::Valid) => {
            for i in 0..input_spatial_shape.len() {
                output_spatial_shape[i] = ((input_spatial_shape[i]
                    - ((attrs.kernel_shape[i] - 1) * dilations[i] + 1) as usize
                    + 1) as f64
                    / strides[i] as f64)
                    .ceil() as usize;
            }
        }
        Some(PoolAutoPad::SameUpper) | Some(PoolAutoPad::SameLower) => {
            for i in 0..input_spatial_shape.len() {
                if attrs.auto_pad == Some(PoolAutoPad::SameUpper) {
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
        Some(PoolAutoPad::NotSet) | None => {}
    }

    attrs.replace_dilations_and_strides(dilations, strides);
    match input_spatial_shape.len() {
        1 => _max_pool_f32_1d(input, attrs, new_pads, output_spatial_shape, output_len),
        2 => _max_pool_f32_2d(input, attrs, new_pads, output_spatial_shape, output_len),
        3 => _max_pool_f32_3d(input, attrs, new_pads, output_spatial_shape, output_len),
        _ => todo!("MaxPool for {}D", input_spatial_shape.len()),
    }
}

fn maxpool_f32(
    input: &ArrayD<f32>,
    attrs: MaxPoolAttrs,
    output_len: usize,
) -> BoxResult<PoolOutput> {
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
        _maxpool_internal_f32(input, attrs, output_len)
    } else {
        _common_pool_f32(input, PoolingType::Max, 0, attrs.into(), output_len)
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_max_pool.py
/// https://onnx.ai/onnx/operators/onnx__MaxPool.html
pub fn maxpool(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64,
    output_len: usize,
) -> BoxResult<OperationResult> {
    if inputs.is_empty() {
        return Err(anyhow!("No inputs"));
    }
    let attrs = MaxPoolAttrs::new(node, inputs[0]);
    match inputs[0] {
        ArrayType::F32(x) => {
            let (y, i) = maxpool_f32(x, attrs, output_len)?;
            Ok((ArrayType::F32(y), i.map(ArrayType::I64)).into())
        }
        _ => todo!("MaxPool for type {}", inputs[0]),
    }
}
