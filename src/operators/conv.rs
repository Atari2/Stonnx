use crate::VERBOSE;
use crate::create_intermediate_output_dir_for;
use crate::named_array_to_file;
use crate::operators::_commonmatmul::matmul_impl;
use itertools::iproduct;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayD;
use ndarray::Ix1;
use ndarray::SliceInfoElem;
use ndarray::{Ix2, IxDyn};

use crate::onnx::AttributeProto;
use crate::onnx::NodeProto;
use crate::utils::ArrayType;
use crate::utils::BoxResult;
use crate::utils::OperationResult;

use anyhow::anyhow;

// Defined but never used because even thought Conv has 2 versions, they both act the same
const _OPSET_VERSIONS: [i64; 2] = [1, 11];

#[derive(Debug, Clone, Copy, PartialEq)]
enum ConvAutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

impl ConvAutoPad {
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

struct ConvAttributes {
    auto_pad: ConvAutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl ConvAttributes {
    fn new_for_recursion(other: &Self) -> Self {
        Self {
            auto_pad: other.auto_pad,
            dilations: other.dilations.clone(),
            group: 1,
            kernel_shape: other.kernel_shape.clone(),
            pads: other.pads.clone(),
            strides: other.strides.clone(),
        }
    }
    fn new(node: &NodeProto, x: &ArrayType, w: &ArrayType) -> Self {
        Self {
            auto_pad: node
                .attribute
                .iter()
                .find(|a| a.name() == "auto_pad")
                .map_or(ConvAutoPad::NotSet, ConvAutoPad::from_attr),
            dilations: node
                .attribute
                .iter()
                .find(|a| a.name() == "dilations")
                .map_or_else(
                    || std::iter::repeat(1_i64).take(x.ndim() - 2).collect(),
                    |a| a.ints.to_vec(),
                ),
            group: node
                .attribute
                .iter()
                .find(|a| a.name() == "group")
                .map_or(1, |a| a.i()),
            kernel_shape: node
                .attribute
                .iter()
                .find(|a| a.name() == "kernel_shape")
                .map_or_else(
                    || w.shape().iter().skip(2).map(|x| *x as i64).collect(),
                    |a| a.ints.to_vec(),
                ),
            pads: node
                .attribute
                .iter()
                .find(|a| a.name() == "pads")
                .map_or_else(
                    || std::iter::repeat(0_i64).take((x.ndim() - 2) * 2).collect(),
                    |a| a.ints.to_vec(),
                ),
            strides: node
                .attribute
                .iter()
                .find(|a| a.name() == "strides")
                .map_or_else(
                    || std::iter::repeat(1_i64).take(x.ndim() - 2).collect(),
                    |a| a.ints.to_vec(),
                ),
        }
    }
}

fn _conv_fast_impl(
    x: ndarray::ArrayViewD<f32>,
    w: ndarray::ArrayViewD<f32>,
    b: Option<ndarray::ArrayViewD<f32>>,
    attrs: ConvAttributes,
) -> BoxResult<ArrayType> {
    create_intermediate_output_dir_for!(conv);
    let dilations = &attrs.dilations;
    let mut kernel_shape = attrs.kernel_shape.clone();
    let strides = &attrs.strides;
    let mut pads = attrs.pads.clone();
    let x_shape = x.shape();
    let w_shape = w.shape();
    let group = attrs.group as usize;
    let mut w = w.to_owned();
    named_array_to_file!(conv, w);
    named_array_to_file!(conv, x);
    if let Some(ref b) = b {
        named_array_to_file!(conv, b);
    }

    if x_shape[1] != w_shape[1] * group || w_shape[0] % group != 0 {
        return Err(anyhow!(
            "Shape inconsistency {} != {} || {} % {} != 0",
            x_shape[1],
            w_shape[1] * group,
            w_shape[0],
            attrs.group
        ));
    }
    if group > 1 {
        let mut res = vec![];
        let mut td = 0;
        let mg = w_shape[0] / group;
        let dw = w_shape[1];

        for b in 0..x_shape[0] {
            for g in 0..group {
                let mut xslicevec = vec![];
                let mut wslicevec = vec![];
                for i in 0..x.ndim() {
                    if i == 0 {
                        xslicevec.push(SliceInfoElem::Slice {
                            start: b as isize,
                            end: Some((b + 1) as isize),
                            step: 1,
                        });
                    } else if i == 1 {
                        xslicevec.push(SliceInfoElem::Slice {
                            start: (g * dw) as isize,
                            end: Some(((g + 1) * dw) as isize),
                            step: 1,
                        });
                    } else {
                        xslicevec.push(SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        });
                    }
                }
                for i in 0..w.ndim() {
                    if i == 0 {
                        wslicevec.push(SliceInfoElem::Slice {
                            start: (g * mg) as isize,
                            end: Some(((g + 1) * mg) as isize),
                            step: 1,
                        });
                    } else {
                        wslicevec.push(SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        });
                    }
                }
                let gx = x.slice(xslicevec.as_slice());
                let gw = w.slice(wslicevec.as_slice());
                let cv = _conv_fast_impl(gx, gw, None, ConvAttributes::new_for_recursion(&attrs))?;
                if b == 0 {
                    td += cv.shape()[1];
                }
                res.push((b, cv));
            }
        }
        let mut new_shape = vec![x_shape[0]];
        new_shape.extend(res[0].1.shape()[1..].iter());
        new_shape[1] = td;
        let mut final_ = ArrayD::<f32>::zeros(new_shape);
        let mut p = 0;
        for (b, cv) in res {
            let mut fslicevec = vec![];
            for i in 0..final_.ndim() {
                if i == 0 {
                    fslicevec.push(SliceInfoElem::Slice {
                        start: b as isize,
                        end: Some((b + 1) as isize),
                        step: 1,
                    });
                } else if i == 1 {
                    fslicevec.push(SliceInfoElem::Slice {
                        start: p as isize,
                        end: Some((p + cv.shape()[1]) as isize),
                        step: 1,
                    });
                } else {
                    fslicevec.push(SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    });
                }
            }
            let cv = if let ArrayType::F32(c) = cv {
                c
            } else {
                return Err(anyhow!("Conv on f32 returned another type"));
            };
            final_.slice_mut(fslicevec.as_slice()).assign(&cv);
            p += cv.shape()[1];
            if p >= final_.shape()[1] {
                p = 0;
            }
        }
        if let Some(b) = b {
            let mut new_shape = vec![1; final_.ndim()];
            new_shape[1] = b.shape()[0];
            let b = b.to_shape(IxDyn(&new_shape))?;
            final_ += &b;
        }
        return Ok(ArrayType::F32(final_));
    }
    if dilations[0] != 1 || dilations.iter().min() != dilations.iter().max() {
        let nd = dilations.len();
        let mut new_kernel_shape = vec![];
        let mut new_shape = w_shape[..w_shape.len() - nd].to_vec();
        for (i, d) in dilations.iter().enumerate() {
            let di = w_shape.len() - nd + i;
            new_shape.push(w_shape[di] + (w_shape[di] - 1) * (d - 1) as usize);
            new_kernel_shape.push(kernel_shape[i] + (kernel_shape[i] - 1) * (d - 1));
        }
        let mut new_w = ArrayD::<f32>::zeros(new_shape);
        let mut indices = vec![
            SliceInfoElem::Slice {
                start: 0,
                end: Some(new_w.shape()[0] as isize),
                step: 1,
            },
            SliceInfoElem::Slice {
                start: 0,
                end: Some(new_w.shape()[1] as isize),
                step: 1,
            },
        ];
        for (i, d) in dilations.iter().enumerate() {
            let di = w_shape.len() - nd + i;
            let slice_idx = new_w.shape()[di];
            indices.push(SliceInfoElem::Slice {
                start: 0,
                end: Some(slice_idx as isize),
                step: *d as isize,
            });
        }
        new_w.slice_mut(indices.as_slice()).assign(&w);
        w = new_w;
        kernel_shape = new_kernel_shape;
    }

    if attrs.auto_pad != ConvAutoPad::NotSet {
        let mut head = vec![];
        let mut tail = vec![];
        for i in 0..(x.ndim() - 2) {
            let d = x_shape[i];
            let target_size = (d + strides[i] as usize - 1) / strides[i] as usize;
            let pad_needed = (target_size - 1) * strides[i] as usize + kernel_shape[i] as usize - d;
            let pad_head = if attrs.auto_pad == ConvAutoPad::SameLower {
                (pad_needed + 1) / 2
            } else {
                pad_needed / 2
            };
            let pad_tail = pad_needed - pad_head;
            head.push(pad_head);
            tail.push(pad_tail);
        }
        pads = head.into_iter().chain(tail).map(|v| v as i64).collect();
    }
    let (c2, mut out_shape) = im2col_fast(&x, &kernel_shape, &pads, strides)?;
    named_array_to_file!(conv, c2);
    let w_reshaped = w.to_shape(vec![
        w.shape().iter().product::<usize>() / c2.shape()[0],
        c2.shape()[0],
    ])?;
    named_array_to_file!(conv, w_reshaped);
    let mut mul = matmul_impl(w_reshaped.view(), c2.view())?;
    named_array_to_file!(conv, mul);
    out_shape.insert(0, w.shape()[0]);
    out_shape.insert(1, x.shape()[0]);
    mul = mul.into_shape(out_shape)?;
    let perm: Vec<usize> = vec![1, 0]
        .into_iter()
        .chain((0..x_shape.len() - 2).map(|x| x + 2))
        .collect();
    mul = mul.permuted_axes(perm);
    named_array_to_file!(conv, mul, "permuted_mul");

    if let Some(b) = b {
        if b.len() == 1 {
            Ok(ArrayType::F32(mul + b))
        } else {
            let mut new_shape = vec![1; mul.ndim()];
            new_shape[1] = b.shape()[0];
            let b = b.to_shape(IxDyn(&new_shape))?;
            named_array_to_file!(conv, b, "reshaped_b");
            let output = mul + &b;
            named_array_to_file!(conv, output);
            Ok(ArrayType::F32(output))
        }
    } else {
        Ok(ArrayType::F32(mul))
    }
}

fn _make_ind(dim: usize, shape: &[i64]) -> BoxResult<ArrayD<i64>> {
    let mut res = ArrayD::<i64>::zeros(shape.iter().map(|x| *x as usize).collect::<Vec<_>>());
    let mut indices = vec![];
    for end in shape.iter() {
        indices.push(SliceInfoElem::Slice {
            start: 0,
            end: Some(*end as isize),
            step: 1,
        });
    }
    let mut new_shape = vec![1_usize; shape.len()];
    new_shape[dim] = shape[dim] as usize;
    let a = Array1::<f32>::range(0.0, shape[dim] as f32, 1.0).mapv(|v| v as i64);
    let first = a.to_shape(IxDyn(&new_shape))?;
    res.slice_mut(indices.as_slice()).assign(&first.view());
    Ok(res)
}

fn im2col_fast(
    x: &ndarray::ArrayViewD<f32>,
    kernel_shape: &[i64],
    pads: &[i64],
    strides: &[i64],
) -> BoxResult<(ArrayD<f32>, Vec<usize>)> {
    let n_dims = kernel_shape.len();
    let (m, n_c) = (x.shape()[0], x.shape()[1]);
    let kernel_size = kernel_shape.iter().product::<i64>() as usize;
    let mut shape_out = vec![];
    for (i, dim) in kernel_shape.iter().enumerate() {
        let dx = x.shape()[i + 2];
        shape_out.push((dx as i64 + pads[i] + pads[i + n_dims] - dim) / strides[i] + 1);
    }
    let mut indices = vec![];
    #[allow(clippy::needless_range_loop)]
    for i in 0..shape_out.len() {
        let kind = _make_ind(i, kernel_shape)?;
        let iind = _make_ind(i, &shape_out)? * strides[i];
        // index = np.tile(kind.ravel(), n_C).reshape(-1, 1) + iind.reshape(1, -1)
        let kindravel = kind.to_shape(Ix1(kind.len()))?.to_owned();
        let res = std::iter::repeat(kindravel)
            .take(n_c)
            .flatten()
            .collect::<Vec<_>>();
        let index = Array2::<i64>::from_shape_vec((res.len(), 1), res)?
            + iind.to_shape(Ix2(1, iind.len()))?.to_owned();
        indices.push(index);
    }
    for (i, index) in indices.iter().enumerate() {
        named_array_to_file!(conv, index, format!("index_{}", i));
    }
    let d = Array2::<i64>::from_shape_vec(
        (n_c * kernel_size, 1),
        (0..n_c)
            .flat_map(|x| std::iter::repeat(x as i64).take(kernel_size))
            .collect::<Vec<_>>(),
    )?;
    named_array_to_file!(conv, d);
    let nc = [[0, 0], [0, 0]];
    let padding = nc
        .iter()
        .copied()
        .chain((0..n_dims).map(|i| [pads[i] as usize, pads[i + n_dims] as usize]))
        .collect::<Vec<_>>();
    let x_padded = ndarray_ndimage::pad(
        x,
        padding.as_slice(),
        ndarray_ndimage::PadMode::Constant(0.0),
    );
    named_array_to_file!(conv, x_padded);

    if !indices.is_empty() {
        let indices_shapes_equal = indices.iter().all(|x| x.shape() == indices[0].shape());
        if !indices_shapes_equal {
            return Err(anyhow!("Indices shapes are not similar"));
        }
        if indices[0].shape()[0] != d.shape()[0] {
            return Err(anyhow!("Indices and d shapes are not broadcastable"));
        }
    }

    // d is always (x, 1)
    // indices are all (x, y)
    // m is x_shape[0] before padding, always first dimension of result
    // result seems to have shape (m, x, y, x_padded.shape()[indices.len() + 1:])
    let rows = indices[0].shape()[0];
    let cols = indices[0].shape()[1];
    let mut cols_shape = vec![m, rows, cols];
    if (indices.len() + 1) < x_padded.ndim() {
        cols_shape.extend(x_padded.shape()[(indices.len() + 2)..].iter());
    }
    let getitem = iproduct!(0..rows, 0..cols).map(|(r, c)| {
        (
            r,
            c,
            [d[[r, 0]] as usize]
                .into_iter()
                .chain(indices.iter().map(move |x| x[[r, c]] as usize)),
        )
    });
    /*
    getitem = (slice(0, m), d, *indices)
    cols = X_padded[getitem]  # type: ignore[index]
     */
    let mut cols = ArrayD::<f32>::zeros(cols_shape.as_slice());
    for i in 0..m {
        for (j, k, index) in getitem.clone() {
            let index = [i].into_iter().chain(index).collect::<Vec<_>>();
            cols[[i, j, k]] += x_padded[index.as_slice()];
        }
    }
    named_array_to_file!(conv, cols);
    let first_dim_len = cols.shape()[0];
    let mut concat_shape = cols.shape()[1..].to_vec();
    let last_dim_len = concat_shape[concat_shape.len() - 1];
    if let Some(last) = concat_shape.last_mut() {
        *last = first_dim_len * last_dim_len;
    }
    let conc_cols = cols.to_shape(concat_shape)?.to_owned().into_dyn();
    named_array_to_file!(conv, conc_cols);
    Ok((
        conc_cols,
        shape_out.iter().map(|v| *v as usize).collect::<Vec<_>>(),
    ))
}

fn conv_fast_impl(
    x: &ArrayType,
    w: &ArrayType,
    b: Option<&ArrayType>,
    attrs: ConvAttributes,
) -> BoxResult<ArrayType> {
    if x.ndim() < 3 {
        return Err(anyhow!("X must have at least 3 dimensions"));
    }
    match (x, w, b) {
        (ArrayType::F32(x), ArrayType::F32(w), Some(ArrayType::F32(b))) => {
            return _conv_fast_impl(x.view(), w.view(), Some(b.view()), attrs);
        }
        (ArrayType::F32(x), ArrayType::F32(w), None) => {
            return _conv_fast_impl(x.view(), w.view(), None, attrs);
        }
        (x, w, b) => {
            if let Some(b) = b {
                todo!("Conv not implemented for types {}, {}, {}", x, w, b);
            } else {
                todo!("Conv not implemented for types {}, {}, None", x, w);
            }
        }
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_conv.py
/// https://onnx.ai/onnx/operators/onnx__Conv.html
pub fn conv(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64, // defined but never used because even thought Conv has 2 versions they both do the same thing
    _output_len: usize,
) -> BoxResult<OperationResult> {
    match inputs.len() {
        2 => {
            let attributes = ConvAttributes::new(node, inputs[0], inputs[1]);
            Ok(conv_fast_impl(inputs[0], inputs[1], None, attributes)?.into())
        }
        3 => {
            let attributes = ConvAttributes::new(node, inputs[0], inputs[1]);
            Ok(conv_fast_impl(inputs[0], inputs[1], Some(inputs[2]), attributes)?.into())
        }
        _ => {
            panic!("Unexpected error: convolution has {} inputs", inputs.len());
        }
    }
}
