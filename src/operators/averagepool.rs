use crate::{
    common::{ArrayElement, ArrayType, BoxResult, F32IntoType, OperationResult},
    onnx::NodeProto,
};
use anyhow::anyhow;
use itertools::Itertools;
use ndarray::{Array0, Array1, ArrayD, Ix0, SliceInfoElem};
use num::traits::AsPrimitive;

use super::_commonpool::{CommonPoolAttrs, PoolAutoPad, PoolOutput};

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

fn get_output_shape_auto_pad(
    auto_pad: PoolAutoPad,
    input_spatial_shape: &[usize],
    kernel_spatial_shape: &[i64],
    strides_spatial: &Option<Vec<i64>>,
) -> BoxResult<Vec<i64>> {
    let strides_spatial = strides_spatial
        .clone()
        .unwrap_or_else(|| vec![1; input_spatial_shape.len()]);
    let mut out_shape = vec![0; input_spatial_shape.len()];
    for i in 0..input_spatial_shape.len() {
        if auto_pad == PoolAutoPad::SameUpper || auto_pad == PoolAutoPad::SameLower {
            out_shape[i] = ((input_spatial_shape[i] as i64 - 1) as f64 / strides_spatial[i] as f64)
                .floor() as i64
                + 1;
        } else if auto_pad == PoolAutoPad::Valid {
            out_shape[i] = ((input_spatial_shape[i] as i64 - kernel_spatial_shape[i]) as f64
                / strides_spatial[i] as f64)
                .floor() as i64
                + 1;
        } else {
            return Err(anyhow!(
                "auto_pad can only be NOTSET, SAME_UPPER, SAME_LOWER, VALID"
            ));
        }
    }
    Ok(out_shape)
}

fn get_pad_shape(
    auto_pad: PoolAutoPad,
    input_spatial_shape: &[usize],
    kernel_spatial_shape: &[i64],
    strides_spatial: &Option<Vec<i64>>,
    output_spatial_shape: &[i64],
) -> BoxResult<Vec<i64>> {
    let spatial_dims = input_spatial_shape.len();
    let mut pad_shape = vec![0; spatial_dims];
    let strides_spatial = strides_spatial
        .clone()
        .unwrap_or_else(|| vec![1; spatial_dims]);
    if auto_pad == PoolAutoPad::SameLower || auto_pad == PoolAutoPad::SameUpper {
        for i in 0..spatial_dims {
            let residual = (output_spatial_shape[i] - 1) * strides_spatial[i]
                + kernel_spatial_shape[i]
                - input_spatial_shape[i] as i64;
            pad_shape[i] = residual;
        }
    } else if auto_pad != PoolAutoPad::Valid {
        return Err(anyhow!(
            "auto_pad can only be NOTSET, SAME_UPPER, SAME_LOWER, VALID"
        ));
    }
    Ok(pad_shape)
}

fn get_pad_with_auto_pad(auto_pad: PoolAutoPad, pad_shape: Vec<i64>) -> Vec<i64> {
    let spatial_dims = pad_shape.len();
    match auto_pad {
        PoolAutoPad::SameUpper => (0..spatial_dims)
            .map(|i| pad_shape[i] / 2)
            .chain((0..spatial_dims).map(|i| pad_shape[i] - pad_shape[i] / 2))
            .collect(),
        PoolAutoPad::SameLower => (0..spatial_dims)
            .map(|i| pad_shape[i] - pad_shape[i] / 2)
            .chain((0..spatial_dims).map(|i| pad_shape[i] / 2))
            .collect(),
        PoolAutoPad::NotSet | PoolAutoPad::Valid => {
            vec![0; spatial_dims * 2]
        }
    }
}

fn get_output_shape_explicit_padding(
    pads: &Option<Vec<i64>>,
    input_spatial_shape: &[usize],
    kernel_spatial_shape: &[i64],
    strides_spatial: &Option<Vec<i64>>,
    dilations: &Option<Vec<i64>>,
    ceil_mode: bool,
) -> BoxResult<(Vec<i64>, Vec<i64>)> {
    let mut output_spatial_shape = vec![0; input_spatial_shape.len()];
    let pads = pads
        .clone()
        .unwrap_or_else(|| vec![0; input_spatial_shape.len() * 2]);
    let strides_spatial = strides_spatial
        .clone()
        .unwrap_or_else(|| vec![1; input_spatial_shape.len()]);
    let dims = input_spatial_shape.len();
    let dilations = Array1::<i64>::from_vec(dilations.clone().unwrap_or_else(|| vec![1; dims]));
    for dim in 0..dims {
        let dim_size = (input_spatial_shape[dim] as i64 + pads[dim] + pads[dims + dim]
            - dilations[dim] * (kernel_spatial_shape[dim] - 1)
            - 1) as f64
            / strides_spatial[dim] as f64
            + 1f64;
        if ceil_mode {
            output_spatial_shape[dim] = dim_size.ceil() as i64;
        } else {
            output_spatial_shape[dim] = dim_size.floor() as i64;
        }
    }
    let mut pads_spatial_shape_new = pads.clone();
    for dim in 0..dims {
        let sliding_window_size = (kernel_spatial_shape[dim] - 1) * dilations[dim] + 1;
        let actual_padded_input_size =
            (output_spatial_shape[dim] - 1) * strides_spatial[dim] + sliding_window_size;
        let extra_pad = actual_padded_input_size
            - input_spatial_shape[dim] as i64
            - pads[dim]
            - pads[dims + dim];
        if extra_pad > 0 {
            pads_spatial_shape_new[dim] += extra_pad / 2;
            pads_spatial_shape_new[dims + dim] += extra_pad - extra_pad / 2;
        }
    }
    Ok((output_spatial_shape, pads_spatial_shape_new))
}

#[allow(clippy::too_many_arguments)]
fn pool<A: ArrayElement>(
    padded: &ArrayD<A>,
    x_shape: &[usize],
    kernel: &[i64],
    strides: &Option<Vec<i64>>,
    out_shape: &[i64],
    pads: &Option<Vec<i64>>,
    dilations: &Option<Vec<i64>>,
    count_include_pad: i64,
) -> BoxResult<PoolOutput<A>>
where
    usize: AsPrimitive<A>,
{
    let spatial_size = x_shape.len() - 2;
    let y_shape = [x_shape[0], x_shape[1]]
        .into_iter()
        .chain(out_shape.iter().map(|v| *v as usize))
        .collect::<Vec<_>>();
    let mut y = ArrayD::<A>::zeros(y_shape);
    let dilations = dilations
        .clone()
        .map_or_else(|| Array1::ones(spatial_size), Array1::from_vec);
    let pads = if let Some(pads) = pads {
        if pads.len() == 1 {
            Array1::<i64>::from_vec(vec![pads[0]; spatial_size * 2])
        } else {
            Array1::<i64>::from_vec(pads.clone())
        }
    } else {
        Array1::<i64>::zeros(spatial_size * 2)
    };
    let strides = strides.clone().unwrap_or_else(|| vec![1; spatial_size]);
    let siter = [0..x_shape[0] as i64, 0..x_shape[1] as i64]
        .into_iter()
        .chain((0..spatial_size).map(|i| {
            let r = (x_shape[i + 2] as i64 + pads[i] + pads[i + spatial_size]
                - (1 + (kernel[i] - 1) * dilations[i]))
                / strides[i]
                + 1;
            0..r
        }));
    for shape in siter.multi_cartesian_product() {
        let slice = [shape[0], shape[1]]
            .into_iter()
            .map(|i| (i as usize).into())
            .chain((0..padded.ndim() - 2).map(|_| (0..).into()))
            .collect::<Vec<SliceInfoElem>>();
        let window = padded.slice(slice.as_slice());
        let window_vals = (0..spatial_size)
            .map(|i| {
                let b = strides[i] * shape[i + 2];
                let e = strides[i] * shape[i + 2] + (1 + (kernel[i] - 1) * dilations[i]);
                (b..e).step_by(dilations[i] as usize)
            })
            .multi_cartesian_product()
            .map(|shape| {
                let shape_len = shape.len();
                let remaining = spatial_size - shape_len;
                let mut slicer = shape
                    .into_iter()
                    .map(|i| (i as usize).into())
                    .collect::<Vec<SliceInfoElem>>();
                if remaining > 0 {
                    slicer.extend((0..remaining).map(|_| SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }));
                }
                window.slice(slicer.as_slice())
            })
            .collect_vec();
        let window_vals_shape = [window_vals.len()]
            .into_iter()
            .chain(window_vals[0].shape().iter().copied())
            .collect::<Vec<_>>();
        let window_vals = ArrayD::<A>::from_shape_vec(
            window_vals_shape,
            window_vals
                .into_iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        )?;
        let shape = shape
            .into_iter()
            .map(|i| (i as usize).into())
            .collect::<Vec<SliceInfoElem>>();
        if count_include_pad == 1 {
            let avg = Array0::from_elem(
                Ix0(),
                window_vals.iter().copied().sum::<A>() / window_vals.len().as_(),
            );
            y.slice_mut(shape.as_slice()).assign(&avg);
        } else {
            let window_vals_no_nan = window_vals
                .into_iter()
                .filter(|v| !v.is_nan())
                .collect::<Vec<_>>();
            let avg = Array0::from_elem(
                Ix0(),
                window_vals_no_nan.iter().copied().sum::<A>() / window_vals_no_nan.len().as_(),
            );
            y.slice_mut(shape.as_slice()).assign(&avg);
        }
    }
    Ok((y, None))
}

fn _average_pool_generic<A: ArrayElement>(
    input: &ArrayD<A>,
    count_include_pad: i64,
    attrs: AveragePoolAttrs,
    _output_len: usize,
) -> BoxResult<PoolOutput<A>>
where
    usize: AsPrimitive<A>,
    f32: F32IntoType<A>,
{
    let x_shape = input.shape();
    let padding_value = if count_include_pad == 0 {
        F32IntoType::as_(f32::NAN)
    } else {
        F32IntoType::as_(0.0)
    };
    match attrs.auto_pad {
        Some(PoolAutoPad::SameUpper) | Some(PoolAutoPad::SameLower) | Some(PoolAutoPad::Valid) => {
            let auto_pad = attrs.auto_pad.unwrap(); // safe
            if attrs.ceil_mode {
                return Err(anyhow!("ceil_mode is not supported with auto_pad"));
            }
            let out_shape = get_output_shape_auto_pad(
                auto_pad,
                &x_shape[2..],
                &attrs.kernel_shape,
                &attrs.strides,
            )?;
            let pads_shape = get_pad_shape(
                auto_pad,
                &x_shape[2..],
                &attrs.kernel_shape,
                &attrs.strides,
                &out_shape,
            )?;
            let pads = get_pad_with_auto_pad(auto_pad, pads_shape);
            let n_dims = pads.len();
            let pads_np = [[0, 0], [0, 0]]
                .into_iter()
                .chain((0..n_dims).map(|i| [pads[i] as usize, pads[i + n_dims] as usize]))
                .collect::<Vec<_>>();
            let padded = ndarray_ndimage::pad(
                input,
                &pads_np,
                ndarray_ndimage::PadMode::Constant(padding_value),
            );
            pool(
                &padded,
                x_shape,
                &attrs.kernel_shape,
                &attrs.strides,
                &out_shape,
                &attrs.pads,
                &attrs.dilations,
                count_include_pad,
            )
        }
        Some(PoolAutoPad::NotSet) | None => {
            let (out_shape, pads) = get_output_shape_explicit_padding(
                &attrs.pads,
                &x_shape[2..],
                &attrs.kernel_shape,
                &attrs.strides,
                &attrs.dilations,
                attrs.ceil_mode,
            )?;
            let n_dims = pads.len() / 2;
            let pads_np = [[0, 0], [0, 0]]
                .into_iter()
                .chain((0..n_dims).map(|i| [pads[i] as usize, pads[i + n_dims] as usize]))
                .collect::<Vec<_>>();
            let padded = ndarray_ndimage::pad(
                input,
                &pads_np,
                ndarray_ndimage::PadMode::Constant(padding_value),
            );
            pool(
                &padded,
                x_shape,
                &attrs.kernel_shape,
                &attrs.strides,
                &out_shape,
                &attrs.pads,
                &attrs.dilations,
                count_include_pad,
            )
        }
    }
}

/// <https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_average_pool.py>
/// <https://onnx.ai/onnx/operators/onnx__AveragePool.html>
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
            let (y, i) =
                _average_pool_generic(x, attrs.count_include_pad as i64, attrs, output_len)?;
            Ok((ArrayType::F32(y), i.map(ArrayType::I64)).into())
        }
        _ => todo!("AveragePool for type {:?}", inputs[0]),
    }
}
