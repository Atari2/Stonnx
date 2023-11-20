use itertools::iproduct;
use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayD;
use ndarray::Axis;
use ndarray::Ix1;
use ndarray::Order;
use ndarray::SliceInfoElem;
use ndarray::{Ix2, IxDyn};

use crate::onnx::AttributeProto;
use crate::onnx::NodeProto;
use crate::utils::shape_safe_product;
use crate::utils::ArrayType;
use crate::utils::BoxResult;
use crate::utils::OperationResult;

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

fn _conv_impl(
    x: ndarray::ArrayViewD<f32>,
    w: ndarray::ArrayViewD<f32>,
    b: Option<ndarray::ArrayViewD<f32>>,
    mut attrs: ConvAttributes,
) -> BoxResult<ArrayType> {
    let x_shape = x.shape();
    let w_shape = w.shape();
    let mut w = w.clone().into_owned();

    let num_channels = x_shape[1];
    let num_feature_maps = w_shape[0];
    let num_input_channels = w_shape[1];
    let group = attrs.group as usize;
    if num_channels != (num_input_channels * group) || num_feature_maps % group != 0 {
        return Err(format!(
            "Shape inconsistency {} != {} || {} % {} != 0",
            num_channels,
            num_input_channels * group,
            num_feature_maps,
            group
        )
        .into());
    }

    if group > 1 {
        let mut res = vec![];
        let mut td = 0;
        let mg = num_feature_maps / group;
        let dw = num_input_channels;

        for b in 0..x_shape[0] {
            for g in 0..group {
                let gx = x.slice(s![b..b + 1, g * dw..(g + 1) * dw, .., ..]);
                let gw = w.slice(s![g * mg..(g + 1) * mg, .., .., ..]);
                let agx = gx.into_dimensionality::<IxDyn>().unwrap();
                let agw = gw.into_dimensionality::<IxDyn>().unwrap();
                let cv = match _conv_impl(agx, agw, None, ConvAttributes::new_for_recursion(&attrs))
                {
                    Ok(ArrayType::F32(v)) => v,
                    Ok(_) => return Err("Conv on f32 returned another type".into()),
                    Err(e) => return Err(e),
                };

                if b == 0 {
                    td += cv.shape()[1];
                }
                res.push((b, cv));
            }
        }
        let mut new_shape = vec![x_shape[0]];
        new_shape.extend(res[0].1.shape().iter().skip(1));
        new_shape[1] = td;
        let mut final_ = ndarray::Array::from_shape_simple_fn(new_shape, || 0_f32);
        let mut p = 0;
        for (b, cv) in res.iter() {
            final_
                .slice_mut(s![*b..*b + 1, p..p + cv.shape()[1], .., ..])
                .assign(cv);
            p += cv.shape()[1];
            if p >= final_.shape()[1] {
                p = 0;
            }
        }
        if let Some(b) = b {
            let b = b.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
            new_shape = std::iter::repeat(1).take(final_.shape().len()).collect();
            new_shape[1] = b.shape()[0];
            let sb = b.into_shape(IxDyn(&new_shape)).unwrap();
            final_ += &sb;
            return Ok(ArrayType::F32(final_));
        }
    }

    let dilations = &attrs.dilations;
    if dilations[0] != 1 || dilations.iter().min() != dilations.iter().max() {
        let nd = dilations.len();
        let mut new_kernel_shape = vec![];
        let mut new_shape: Vec<usize> = w_shape.iter().take(w_shape.len() - nd).copied().collect();
        for (i, d) in dilations.iter().enumerate() {
            let di = w_shape.len() - nd + i;
            new_shape.push(w_shape[di] + (w_shape[di] - 1) * (d - 1) as usize);
            new_kernel_shape.push(attrs.kernel_shape[i] + (attrs.kernel_shape[i] - 1) * (d - 1));
        }
        let mut new_w = ndarray::Array::from_shape_fn(new_shape, |_| 0f32);
        let mut indices = vec![s![0..new_w.shape()[0]], s![0..new_w.shape()[1]]];
        for (i, d) in dilations.iter().enumerate() {
            let di = w_shape.len() - nd + i;
            let slice_idx = new_w.shape()[di];
            indices.push(s![0..slice_idx; *d as usize]);
        }
        for s in indices.iter() {
            new_w.slice_mut(s).assign(&w);
        }
        w = new_w;
        attrs.kernel_shape = new_kernel_shape;
    }

    if attrs.auto_pad != ConvAutoPad::NotSet {
        let mut head = vec![];
        let mut tail = vec![];
        for (i, item) in x_shape.iter().enumerate().take(x_shape.len() - 2) {
            let d = *item as i64;
            let target_size = (d + attrs.strides[i] - 1) / attrs.strides[i];
            let pad_needed = (target_size - 1) * attrs.strides[i] + attrs.kernel_shape[i] - d;
            let pad_head = if attrs.auto_pad == ConvAutoPad::SameLower {
                (pad_needed + 1) / 2
            } else {
                pad_needed / 2
            };
            let pad_tail = pad_needed - pad_head;
            head.push(pad_head);
            tail.push(pad_tail);
        }
        attrs.pads = head.into_iter().chain(tail).collect();
    }

    if x_shape.len() == 3 {
        let (s_n, s_c, s_h) = (x_shape[0] as i64, x_shape[1] as i64, x_shape[2] as i64);
        let kh = attrs.kernel_shape[0];
        let sth = attrs.strides[0];

        let h_out = ((s_h - kh + attrs.pads[0] + attrs.pads[1]) / sth) + 1;

        let h0 = attrs.pads[0];
        let oh = -(kh % 2);
        let bh = -h0;
        let eh = h_out * sth;
        let mut res = ndarray::Array::from_shape_simple_fn(
            vec![x_shape[0], w_shape[0], h_out as usize],
            || 0_f32,
        );
        if let Some(b) = b {
            let b = b.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
            let sb = b.into_shape((1, b.shape()[0], 1)).unwrap();
            let mut mutview = res.slice_mut(s![.., .., ..]);
            mutview += &sb;
        }
        for n in 0..s_n as usize {
            for nw in 0..w_shape[0] {
                for c in 0..s_c as usize {
                    // w = W[nw : nw + 1, c : c + 1]
                    let w = w.slice(s![nw..nw + 1, c..c + 1, .., ..]);
                    for io in (bh..eh).step_by(sth as usize) {
                        let hr = (io - bh) / sth;
                        if hr >= h_out {
                            continue;
                        }
                        let i = io + kh % 2;
                        let (ih1, ih2) = (0.max(i + oh), (i + oh + kh).min(s_h));
                        let img = x.slice(s![n..n + 1, c..c + 1, ih1 as usize..ih2 as usize, ..]);
                        let s = if img.shape() != w_shape {
                            let (jh1, jh2) = ((-oh - i).max(0), kh.min(kh + s_h - (i + oh + kh)));
                            let w_ = w.slice(s![..1, ..1, jh1 as usize..jh2 as usize, ..]);
                            if img.shape() != w_.shape() {
                                return Err("Shape unexpected".into());
                            }
                            let imgs = img.into_shape((1, shape_safe_product(img.shape()))).unwrap();
                            let w_s = w_.into_shape((shape_safe_product(w_.shape()), 1)).unwrap();
                            ndarray::ArrayView2::dot(&imgs, &w_s)[[0, 0]]
                        } else {
                            let imgs = img.into_shape((1, shape_safe_product(img.shape()))).unwrap();
                            let ws = w.into_shape((shape_safe_product(w.shape()), 1)).unwrap();
                            ndarray::ArrayView2::dot(&imgs, &ws)[[0, 0]]
                        };
                        res[[n, nw, hr as usize]] += s;
                    }
                }
            }
        }
        return Ok(ArrayType::F32(res));
    }

    if x_shape.len() == 4 {
        let (s_n, s_c, s_h, s_w) = (
            x_shape[0] as i64,
            x_shape[1] as i64,
            x_shape[2] as i64,
            x_shape[3] as i64,
        );
        let (kh, kw) = (attrs.kernel_shape[0], attrs.kernel_shape[1]);
        let (sth, stw) = (attrs.strides[0], attrs.strides[1]);

        let h_out = ((s_h - kh + attrs.pads[0] + attrs.pads[2]) / sth) + 1;
        let w_out = ((s_w - kw + attrs.pads[1] + attrs.pads[3]) / stw) + 1;

        let (h0, w0) = (attrs.pads[0], attrs.pads[1]);
        let (oh, ow) = (-(kh % 2), -(kw % 2));
        let (bh, bw) = (-h0, -w0);
        let (eh, ew) = (h_out * sth, w_out * stw);
        let mut res = ndarray::Array::from_shape_simple_fn(
            vec![x_shape[0], w_shape[0], h_out as usize, w_out as usize],
            || 0_f32,
        );
        if let Some(b) = b {
            let b = b.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
            let sb = b.into_shape((1, b.shape()[0], 1, 1)).unwrap();
            let mut mutview = res.slice_mut(s![.., .., .., ..]);
            mutview += &sb;
        }
        for n in 0..s_n as usize {
            for nw in 0..w_shape[0] {
                for c in 0..s_c as usize {
                    // w = W[nw : nw + 1, c : c + 1]
                    let w = w.slice(s![nw..nw + 1, c..c + 1, .., ..]);
                    for io in (bh..eh).step_by(sth as usize) {
                        let hr = (io - bh) / sth;
                        if hr >= h_out {
                            continue;
                        }
                        let i = io + kh % 2;
                        let (ih1, ih2) = (0.max(i + oh), (i + oh + kh).min(s_h));
                        for jo in (bw..ew).step_by(stw as usize) {
                            let wr = (jo - bw) / stw;
                            if wr >= w_out {
                                continue;
                            }
                            let j = jo + kw % 2;
                            let (jw1, jw2) = (0.max(j + ow), (j + ow + kw).min(s_w));
                            let img = x.slice(s![
                                n..n + 1,
                                c..c + 1,
                                ih1 as usize..ih2 as usize,
                                jw1 as usize..jw2 as usize
                            ]);
                            let s = if img.shape() != w_shape {
                                let (jh1, jh2) =
                                    ((-oh - i).max(0), kh.min(kh + s_h - (i + oh + kh)));
                                let (jw1, jw2) =
                                    ((-ow - j).max(0), kw.min(kw + s_w - (j + ow + kw)));
                                let w_ = w.slice(s![
                                    ..1,
                                    ..1,
                                    jh1 as usize..jh2 as usize,
                                    jw1 as usize..jw2 as usize
                                ]);
                                if img.shape() != w_.shape() {
                                    return Err("Shape unexpected".into());
                                }
                                let imgs = img.to_shape((
                                    (1, shape_safe_product(img.shape())),
                                    Order::RowMajor,
                                )).unwrap();
                                let w_s = w_.to_shape((
                                    (shape_safe_product(w_.shape()), 1),
                                    Order::RowMajor,
                                )).unwrap();
                                let imgview = imgs.view();
                                let wview = w_s.view();
                                ndarray::ArrayView2::dot(&imgview, &wview)[[0, 0]]
                            } else {
                                let copied_img = ndarray::Array::from_shape_vec(
                                    img.raw_dim(),
                                    img.iter().cloned().collect(),
                                ).unwrap();
                                let imgs = copied_img
                                    .view()
                                    .into_shape((1, shape_safe_product(img.shape()))).unwrap();
                                let copied_w = ndarray::Array::from_shape_vec(
                                    w.raw_dim(),
                                    w.iter().cloned().collect(),
                                ).unwrap();
                                let ws = copied_w
                                    .view()
                                    .into_shape((shape_safe_product(w.shape()), 1)).unwrap();
                                ndarray::ArrayView2::dot(&imgs, &ws)[[0, 0]]
                            };
                            res[[n, nw, hr as usize, wr as usize]] += s;
                        }
                    }
                }
            }
        }
        return Ok(ArrayType::F32(res));
    }

    if x_shape.len() == 5 {
        todo!("Conv with X 3D not implemented yet, look at https://github.com/onnx/onnx/blob/ab5bdf8d6d77432cf7892ff702d926b8582b2704/onnx/reference/ops/op_conv.py#L215");
    }
    todo!("Conv not implemented for shape {:?}", x_shape);
}

/// CONVOLUTION
/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_conv.py
/// attributes: auto_pad, dilations, group, kernel_shape, pads, strides
/// inputs: X, W, B
///         X: input (N x C x H x W)
///             N: batch size
///             C: number of channels
///             H: height
///             W: width
///         W: weight (M x C/group x kH x kW)
///             M: number of feature maps
///             C/group: number of input channels
///             kH: kernel height
///             kW: kernel width
///         B: bias (M)
/// outputs: Y
///         Y: output
fn conv_impl(
    x: &ArrayType,
    w: &ArrayType,
    b: Option<&ArrayType>,
    attrs: ConvAttributes,
) -> BoxResult<ArrayType> {
    match (x, w, b) {
        (ArrayType::F32(x), ArrayType::F32(w), Some(ArrayType::F32(b))) => {
            return _conv_impl(x.view(), w.view(), Some(b.view()), attrs);
        }
        (ArrayType::F32(x), ArrayType::F32(w), None) => {
            return _conv_impl(x.view(), w.view(), None, attrs);
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

fn matmul(a: ndarray::ArrayViewD<f32>, b: ndarray::ArrayViewD<f32>) -> BoxResult<ArrayD<f32>> {
    use ndarray::linalg::general_mat_mul;
    if a.ndim() == 2 && b.ndim() == 2 {
        let a = a.into_dimensionality::<Ix2>().unwrap();
        let b = b.into_dimensionality::<Ix2>().unwrap();
        let mut c = Array2::<f32>::zeros((a.shape()[0], b.shape()[1]));
        general_mat_mul(1.0, &a, &b, 1.0, &mut c);
        Ok(c.into_dyn())
    } else {
        todo!(
            "Matmul not implemented for ndim {} and {}",
            a.ndim(),
            b.ndim()
        );
    }
}

fn _conv_fast_impl(
    x: ndarray::ArrayViewD<f32>,
    w: ndarray::ArrayViewD<f32>,
    b: Option<ndarray::ArrayViewD<f32>>,
    attrs: ConvAttributes,
) -> BoxResult<ArrayType> {
    let dilations = &attrs.dilations;
    let mut kernel_shape = attrs.kernel_shape.clone();
    let strides = &attrs.strides;
    let mut pads = attrs.pads.clone();
    let x_shape = x.shape();
    let w_shape = w.shape();
    let group = attrs.group as usize;
    let mut w = w.to_owned();

    if x_shape[1] != w_shape[1] * group || w_shape[0] % group != 0 {
        return Err(format!(
            "Shape inconsistency {} != {} || {} % {} != 0",
            x_shape[1],
            w_shape[1] * group,
            w_shape[0],
            attrs.group
        )
        .into());
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
                let cv = _conv_fast_impl(gx, gw, None, ConvAttributes::new_for_recursion(&attrs)).unwrap();
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
                return Err("Conv on f32 returned another type".into());
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
            let b = b.to_shape(IxDyn(&new_shape)).unwrap();
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
    let (c2, mut out_shape) = im2col_fast(&x, &kernel_shape, &pads, strides).unwrap();
    let w_reshaped = w.to_shape(
        vec![w.shape().iter().product::<usize>() / c2.shape()[0],c2.shape()[0]]
    ).unwrap();
    let mut mul = matmul(w_reshaped.view(), c2.view()).unwrap();
    out_shape.insert(0, w.shape()[0]);
    out_shape.insert(1, x.shape()[0]);
    mul = mul.into_shape(out_shape).unwrap();
    let perm: Vec<usize> = vec![1, 0]
        .into_iter()
        .chain((0..x_shape.len() - 2).map(|x| x + 2))
        .collect();
    mul = mul.permuted_axes(perm);

    if let Some(b) = b {
        if b.len() == 1 {
            Ok(ArrayType::F32(mul + b))
        } else {
            let mut new_shape = vec![1; mul.ndim()];
            new_shape[1] = b.shape()[0];
            let b = b.to_shape(IxDyn(&new_shape)).unwrap();
            Ok(ArrayType::F32(mul + b))
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
    let first = a.to_shape(IxDyn(&new_shape)).unwrap();
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
        let kind = _make_ind(i, kernel_shape).unwrap();
        let iind = _make_ind(i, &shape_out).unwrap() * strides[i];
        // index = np.tile(kind.ravel(), n_C).reshape(-1, 1) + iind.reshape(1, -1)
        let kindravel = kind.to_shape(Ix1(kind.len())).unwrap().to_owned();
        let res = std::iter::repeat(kindravel)
            .take(n_c)
            .flatten()
            .collect::<Vec<_>>();
        let index = Array2::<i64>::from_shape_vec((res.len(), 1), res).unwrap()
            + iind.to_shape(Ix2(1, iind.len())).unwrap().to_owned();
        indices.push(index);
    }
    let d = Array2::<i64>::from_shape_vec(
        (n_c * kernel_size, 1),
        std::iter::repeat((0..n_c).map(|i| i as i64))
            .take(kernel_size)
            .flatten()
            .collect::<Vec<_>>(),
    ).unwrap();
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

    if !indices.is_empty() {
        let indices_shapes_equal = indices.iter().all(|x| x.shape() == indices[0].shape());
        if !indices_shapes_equal {
            return Err("Indices shapes are not similar".into());
        }
        if indices[0].shape()[0] != d.shape()[0] {
            return Err("Indices and d shapes are not broadcastable".into());
        }
    }

    // d is always (x, 0)
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
        [d[[r, 0]] as usize].into_iter()
            .chain(indices.iter().map(move |x| x[[r, c]] as usize))
    });
    /*
    getitem = (slice(0, m), d, *indices)
    cols = X_padded[getitem]  # type: ignore[index]
     */
    
    let mut cols = ArrayD::<f32>::zeros(cols_shape.as_slice());
    for index in getitem {
        let mut cols_index = index.collect::<Vec<_>>();
        cols_index[0] = 0;
        let index = [m - 1].into_iter().chain(cols_index.iter().copied()).collect::<Vec<_>>();
        cols[cols_index.as_slice()] = x_padded[index.as_slice()];
    }
    let first_dim_len = cols.shape()[0];
    let mut concat_shape = cols.shape()[1..].to_vec();
    let last_dim_len = concat_shape[concat_shape.len() - 1];
    if let Some(last) = concat_shape.last_mut() {
        *last = first_dim_len * last_dim_len;
    }
    let conc_cols = cols.to_shape(concat_shape).unwrap().to_owned().into_dyn();
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
        return Err("X must have at least 3 dimensions".into());
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
            Ok(conv_fast_impl(inputs[0], inputs[1], None, attributes).unwrap().into())
        }
        3 => {
            let attributes = ConvAttributes::new(node, inputs[0], inputs[1]);
            Ok(conv_fast_impl(inputs[0], inputs[1], Some(inputs[2]), attributes).unwrap().into())
        }
        _ => {
            panic!("Unexpected error: convolution has {} inputs", inputs.len());
        }
    }
}
