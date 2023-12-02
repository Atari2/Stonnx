#![allow(clippy::too_many_arguments)]
use crate::{
    common::{ArrayElement, BoxResult, F32IntoType},
    onnx::AttributeProto,
};
use anyhow::anyhow;
use itertools::Itertools;
use ndarray::{s, Array0, Array1, Array2, ArrayD, ArrayView1, Ix0, Ix2, SliceInfoElem};
use ndarray_stats::QuantileExt;
use num::traits::AsPrimitive;

#[derive(Debug, PartialEq)]
pub enum PoolingType {
    Max,
    Average,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum PoolAutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

impl PoolAutoPad {
    pub fn from_attr(s: &AttributeProto) -> Self {
        match std::str::from_utf8(s.s()) {
            Ok("NOTSET") => Self::NotSet,
            Ok("SAME_UPPER") => Self::SameUpper,
            Ok("SAME_LOWER") => Self::SameLower,
            Ok("VALID") => Self::Valid,
            _ => Self::NotSet,
        }
    }
}

pub type PoolOutput<A> = (ArrayD<A>, Option<ArrayD<i64>>);

#[derive(Debug)]
pub struct CommonPoolAttrs {
    pub auto_pad: Option<PoolAutoPad>,
    pub ceil_mode: bool,
    pub dilations: Option<Vec<i64>>,
    pub kernel_shape: Vec<i64>,
    pub pads: Option<Vec<i64>>,
    pub storage_order: i64,
    pub strides: Option<Vec<i64>>,
}

fn _get_pad_shape(
    auto_pad: PoolAutoPad,
    input_spatial_shape: &[usize],
    kernel_spatial_shape: &[i64],
    strides_spatial: &[i64],
    output_spatial_shape: &[usize],
) -> BoxResult<Vec<usize>> {
    let mut pad_shape = vec![0, input_spatial_shape.len()];
    if auto_pad == PoolAutoPad::SameLower || auto_pad == PoolAutoPad::SameUpper {
        for i in 0..input_spatial_shape.len() {
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] as usize
                + kernel_spatial_shape[i] as usize
                - input_spatial_shape[i];
        }
    }
    if pad_shape.is_empty() {
        Err(anyhow!("Pad shape cannot be empty"))
    } else {
        Ok(pad_shape)
    }
}

fn _get_output_shape_no_ceil(
    auto_pad: Option<PoolAutoPad>,
    input_spatial_shape: &[usize],
    kernel_spatial_shape: &[i64],
    strides_spatial: &[i64],
) -> Vec<usize> {
    let mut out_shape = vec![0; input_spatial_shape.len()];
    if auto_pad == Some(PoolAutoPad::SameUpper) || auto_pad == Some(PoolAutoPad::SameLower) {
        for i in 0..input_spatial_shape.len() {
            out_shape[i] =
                (input_spatial_shape[i] as f32 / strides_spatial[i] as f32).ceil() as usize;
        }
    } else if auto_pad == Some(PoolAutoPad::Valid) {
        for i in 0..input_spatial_shape.len() {
            out_shape[i] = (input_spatial_shape[i] as f32
                - (kernel_spatial_shape[i] - 1) as f32 / strides_spatial[i] as f32)
                .ceil() as usize;
        }
    }
    out_shape
}

fn _get_output_shape(
    auto_pad: Option<PoolAutoPad>,
    input_spatial_shape: &[usize],
    kernel_spatial_shape: &[i64],
    strides_spatial: &[i64],
    pad_shape: Option<&[usize]>,
    ceil_mode: bool,
) -> BoxResult<Vec<usize>> {
    let out_shape = if ceil_mode {
        let mut out_shape = vec![0; input_spatial_shape.len()];
        if auto_pad == Some(PoolAutoPad::SameLower) || auto_pad == Some(PoolAutoPad::SameUpper) {
            for i in 0..input_spatial_shape.len() {
                out_shape[i] =
                    f32::ceil(input_spatial_shape[i] as f32 / strides_spatial[i] as f32) as usize;
            }
        } else if auto_pad == Some(PoolAutoPad::Valid) {
            if let Some(pad_shape) = pad_shape {
                for i in 0..input_spatial_shape.len() {
                    out_shape[i] = f32::ceil(
                        (input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i] as usize)
                            as f32
                            / strides_spatial[i] as f32
                            + 1f32,
                    ) as usize;
                }
            } else {
                return Err(anyhow!(
                    "pad_shape cannot be none if auto_pad is valid and ceil_mode is 1"
                ));
            }
        }
        out_shape
    } else {
        _get_output_shape_no_ceil(
            auto_pad,
            input_spatial_shape,
            kernel_spatial_shape,
            strides_spatial,
        )
    };
    if out_shape.is_empty() {
        Err(anyhow!("Output shape cannot be empty"))
    } else if out_shape.iter().min().copied().unwrap_or(0) == 0 {
        Err(anyhow!("Output shape cannot be less than or equal to 0"))
    } else {
        Ok(out_shape)
    }
}

fn _get_indices(i: usize, shape: &[usize]) -> Array1<usize> {
    let mut res = Array1::zeros(shape.len());
    let mut k = shape.len() - 1;
    let mut i = i as f32;
    while k > 0 {
        let m = i % shape[k] as f32;
        res[k] = m as usize;
        i -= m;
        i /= shape[k] as f32;
        k -= 1;
    }
    res[0] = i as usize;
    res
}

fn _get_index(indices: Array1<usize>, shape: &[usize]) -> usize {
    let mut ind = 0;
    let mut mul = 1;
    for (pos, sh) in indices.iter().rev().zip(shape.iter().rev()) {
        ind += pos * mul;
        mul *= *sh;
    }
    ind
}

fn _pool_generic<A: ArrayElement>(
    padded: &ArrayD<A>,
    x_shape: &[usize],
    kernel_shape: &[i64],
    strides_shape: &[i64],
    out_shape: &[usize],
    pad_shape: &[usize],
    pooling_type: PoolingType,
    count_include_pad: Option<i64>,
    ceil_mode: Option<bool>,
    indices: bool,
    pads: &Array2<usize>,
) -> BoxResult<PoolOutput<A>>
where
    usize: AsPrimitive<A>,
{
    let fpool: fn(ArrayView1<A>) -> Array0<A> = match pooling_type {
        PoolingType::Max => |a: ArrayView1<A>| -> Array0<A> {
            Array0::from_elem(Ix0(), a.iter().copied().fold(A::MIN, A::max))
        },
        PoolingType::Average => |a: ArrayView1<A>| -> Array0<A> {
            Array0::from_elem(Ix0(), a.iter().copied().sum::<A>() / a.len().as_())
        },
    };
    let spatial_size = x_shape.len() - 2;
    let mut y = ndarray::ArrayD::<A>::zeros(
        [x_shape[0], x_shape[1]]
            .into_iter()
            .chain(out_shape.iter().copied())
            .collect::<Vec<_>>(),
    );
    let mut z = if indices {
        Some(ndarray::ArrayD::<i64>::from_elem(y.shape(), -1))
    } else {
        None
    };
    let round_fct = if let Some(true) = ceil_mode {
        f32::ceil
    } else {
        f32::floor
    };
    let loop_range = || {
        let getrange = |i| {
            let div1 = (x_shape[i + 2] + pad_shape[i] - kernel_shape[i] as usize) as f32;
            let div2 = strides_shape[i] as f32;
            let res = div1 / div2 + 1f32;
            0..round_fct(res) as usize
        };
        (0..spatial_size).map(getrange)
    };
    let loop_iter = loop_range();
    let loop_vec = [(0..x_shape[0]), (0..x_shape[1])]
        .into_iter()
        .chain(loop_iter)
        .collect::<Vec<_>>();
    for shape in loop_vec.into_iter().multi_cartesian_product() {
        let shape: Vec<usize> = shape;
        let mut sliceinfos = vec![];
        #[allow(clippy::needless_range_loop)] // bogus warning
        for i in 0..padded.ndim() {
            if i < 2 {
                sliceinfos.push(SliceInfoElem::Index(shape[i] as isize));
            } else {
                sliceinfos.push(SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                });
            }
        }
        let window = padded.slice(sliceinfos.as_slice());
        let listi = (0..spatial_size)
            .map(|i| {
                (strides_shape[i] as usize * shape[i + 2])
                    ..(strides_shape[i] as usize * shape[i + 2] + kernel_shape[i] as usize)
            })
            .collect::<Vec<_>>();
        let listi2 = listi
            .into_iter()
            .multi_cartesian_product()
            .collect::<Vec<_>>();
        let mut values = vec![];
        for iv in listi2 {
            let wsi = (0..window.ndim())
                .map(|j| {
                    if j < iv.len() {
                        SliceInfoElem::Index(iv[j] as isize)
                    } else {
                        SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        }
                    }
                })
                .collect::<Vec<_>>();
            values.push(window.slice(wsi.as_slice()));
        }
        let values = values.iter().flatten().copied().collect::<Vec<_>>();
        let window_vals = ndarray::Array1::<A>::from_vec(values);
        if count_include_pad == Some(1) && pooling_type == PoolingType::Average {
            let shapeidx = (0..y.ndim())
                .map(|i| {
                    if i < shape.len() {
                        SliceInfoElem::Index(shape[i] as isize)
                    } else {
                        SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        }
                    }
                })
                .collect::<Vec<_>>();
            let pooled = fpool(window_vals.view());
            y.slice_mut(shapeidx.as_slice()).assign(&pooled);
        } else {
            let no_nan = window_vals
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .collect::<Vec<_>>();
            let no_nan = ndarray::Array1::<A>::from_vec(no_nan);
            let pooled = fpool(no_nan.view());
            let shapeidx = (0..y.ndim())
                .map(|i| {
                    if i < shape.len() {
                        SliceInfoElem::Index(shape[i] as isize)
                    } else {
                        SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        }
                    }
                })
                .collect::<Vec<_>>();
            y.slice_mut(shapeidx.as_slice()).assign(&pooled);
            if let Some(ref mut z) = z {
                if indices {
                    let arg = no_nan.argmax()?;
                    let coordinates = _get_indices(arg, out_shape);
                    let delta = Array1::from_vec(shape[2..].to_vec()) - pads.slice(s![.., 0]);
                    let coordinates = coordinates + delta;
                    let new_arg = _get_index(coordinates, &x_shape[2..]);
                    z[shape.as_slice()] = new_arg as i64;
                }
            }
        }
    }
    Ok((y, z))
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/_op_common_pool.py
pub fn _commn_pool_generic<A: ArrayElement>(
    input: &ArrayD<A>,
    pooling_type: PoolingType,
    count_include_pad: i64,
    mut attrs: CommonPoolAttrs,
    output_len: usize,
) -> BoxResult<PoolOutput<A>>
where
    f32: F32IntoType<A>,
    usize: AsPrimitive<A>,
{
    if attrs.dilations.is_none() && pooling_type == PoolingType::Max {
        attrs.dilations = Some(vec![1; attrs.kernel_shape.len()]);
    }
    let pads = if let Some(ref pads) = attrs.pads {
        pads.clone()
    } else {
        vec![0; attrs.kernel_shape.len() * 2]
    };
    let strides = if let Some(strides) = attrs.strides {
        if strides.is_empty() {
            vec![1; input.ndim() - 2]
        } else {
            strides
        }
    } else {
        vec![1; input.ndim() - 2]
    };
    let kernel_shape = attrs.kernel_shape;
    let auto_pad = if let Some(auto_pad) = attrs.auto_pad {
        if auto_pad == PoolAutoPad::NotSet {
            Some(PoolAutoPad::Valid)
        } else {
            Some(auto_pad)
        }
    } else {
        None
    };
    let (mut pad_shape, x_shape, mut padded) = if pads.is_empty() {
        (
            vec![0; input.ndim() - 2],
            input.shape()[2..].to_vec(),
            input.clone(),
        )
    } else if pads.len() == 4 {
        let (pad_top, pad_bottom, pad_left, pad_right) = (
            pads[0] as usize,
            pads[1] as usize,
            pads[2] as usize,
            pads[3] as usize,
        );
        let pad_shape = vec![pad_top + pad_bottom, pad_left + pad_right];
        let x_shape = input.shape()[2..]
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| v + pad_shape[i])
            .collect::<Vec<_>>();
        let const_: A = if count_include_pad == 0 {
            F32IntoType::as_(f32::NAN)
        } else {
            F32IntoType::as_(0.0)
        };
        let padded = ndarray_ndimage::pad(
            input,
            &[[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]],
            ndarray_ndimage::PadMode::Constant(const_),
        );
        (pad_shape, x_shape, padded)
    } else {
        (
            pads.iter().map(|v| *v as usize).collect(),
            input.shape()[2..].to_vec(),
            input.clone(),
        )
    };
    let out_shape;
    if auto_pad == Some(PoolAutoPad::SameLower) || auto_pad == Some(PoolAutoPad::SameUpper) {
        let const_: A = if count_include_pad == 0 {
            F32IntoType::as_(f32::NAN)
        } else {
            F32IntoType::as_(0.0)
        };
        out_shape = _get_output_shape(
            auto_pad, // safe to unwrap, verified above
            &x_shape,
            &kernel_shape,
            &strides,
            Some(&pad_shape),
            attrs.ceil_mode,
        )?;
        pad_shape = _get_pad_shape(
            auto_pad.unwrap(),
            &x_shape,
            &kernel_shape,
            &strides,
            &out_shape,
        )?;
        let (pb, pt, pr, pl) = if auto_pad == Some(PoolAutoPad::SameLower) {
            let pb = pad_shape[0] / 2;
            let pr = pad_shape[1] / 2;
            (pb, pad_shape[0] - pb, pr, pad_shape[1] - pr)
        } else {
            let pt = pad_shape[0] / 2;
            let pl = pad_shape[1] / 2;
            (pad_shape[0] - pt, pt, pad_shape[1] - pl, pl)
        };
        padded = ndarray_ndimage::pad(
            &padded,
            &[[0, 0], [0, 0], [pt, pb], [pl, pr]],
            ndarray_ndimage::PadMode::Constant(const_),
        );
    } else {
        out_shape = _get_output_shape(
            auto_pad,
            &x_shape,
            &kernel_shape,
            &strides,
            Some(&pad_shape),
            attrs.ceil_mode,
        )?
    }
    let ndims = pads.len() / 2;
    let new_pads_dim = Ix2(ndims, 2);
    let new_pads_vec = (0..ndims)
        .flat_map(|i| [pads[i] as usize, pads[i + ndims] as usize].into_iter())
        .collect::<Vec<_>>();
    let new_pads = ndarray::Array2::from_shape_vec(new_pads_dim, new_pads_vec)?;
    _pool_generic(
        &padded,
        input.shape(),
        &kernel_shape,
        &strides,
        &out_shape,
        &pad_shape,
        pooling_type,
        Some(count_include_pad),
        Some(attrs.ceil_mode),
        output_len > 1,
        &new_pads,
    )
}
