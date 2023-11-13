use ndarray::s;
use ndarray::IxDyn;

use crate::onnx::AttributeProto;
use crate::onnx::NodeProto;
use crate::utils::ArrayType;

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
                    || std::iter::repeat(1_i64).take(x.shape().len() - 2).collect(),
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
                    || {
                        std::iter::repeat(0_i64)
                            .take((x.shape().len() - 2) * 2)
                            .collect()
                    },
                    |a| a.ints.to_vec(),
                ),
            strides: node
                .attribute
                .iter()
                .find(|a| a.name() == "strides")
                .map_or_else(
                    || std::iter::repeat(1_i64).take(x.shape().len() - 2).collect(),
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
) -> Result<ArrayType, Box<dyn std::error::Error>> {
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
                let agx = gx.into_dimensionality::<IxDyn>()?;
                let agw = gw.into_dimensionality::<IxDyn>()?;
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
            let b = b.clone().into_dimensionality::<ndarray::Ix1>()?;
            new_shape = std::iter::repeat(1).take(final_.shape().len()).collect();
            new_shape[1] = b.shape()[0];
            let sb = b.into_shape(IxDyn(&new_shape))?;
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
            let b = b.clone().into_dimensionality::<ndarray::Ix1>()?;
            let sb = b.into_shape((1, b.shape()[0], 1))?;
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
                        let (ih1, ih2) =
                            (std::cmp::max(0, i + oh), std::cmp::min(i + oh + kh, s_h));
                        let img = x.slice(s![n..n + 1, c..c + 1, ih1 as usize..ih2 as usize, ..]);
                        let s = if img.shape() != w_shape {
                            let (jh1, jh2) = (
                                std::cmp::max(-oh - i, 0),
                                std::cmp::min(kh, kh + s_h - (i + oh + kh)),
                            );
                            let w_ = w.slice(s![..1, ..1, jh1 as usize..jh2 as usize, ..]);
                            if img.shape() != w_.shape() {
                                return Err("Shape unexpected".into());
                            }
                            let imgs = img.into_shape((1, img.shape().iter().product()))?;
                            let w_s = w_.into_shape((w_.shape().iter().product(), 1))?;
                            ndarray::ArrayView2::dot(&imgs, &w_s)[[0, 0]]
                        } else {
                            let imgs = img.into_shape((1, img.shape().iter().product()))?;
                            let ws = w.into_shape((w.shape().iter().product(), 1))?;
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
            let b = b.clone().into_dimensionality::<ndarray::Ix1>()?;
            let sb = b.into_shape((1, b.shape()[0], 1, 1))?;
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
                        let (ih1, ih2) =
                            (std::cmp::max(0, i + oh), std::cmp::min(i + oh + kh, s_h));
                        for jo in (bw..ew).step_by(stw as usize) {
                            let wr = (jo - bw) / stw;
                            if wr >= w_out {
                                continue;
                            }
                            let j = jo + kw % 2;
                            let (jw1, jw2) =
                                (std::cmp::max(0, j + ow), std::cmp::min(j + ow + kw, s_w));
                            let img = x.slice(s![
                                n..n + 1,
                                c..c + 1,
                                ih1 as usize..ih2 as usize,
                                jw1 as usize..jw2 as usize
                            ]);
                            let s = if img.shape() != w_shape {
                                let (jh1, jh2) = (
                                    std::cmp::max(-oh - i, 0),
                                    std::cmp::min(kh, kh + s_h - (i + oh + kh)),
                                );
                                let (jw1, jw2) = (
                                    std::cmp::max(-ow - j, 0),
                                    std::cmp::min(kw, kw + s_w - (j + ow + kw)),
                                );
                                let w_ = w.slice(s![
                                    ..1,
                                    ..1,
                                    jh1 as usize..jh2 as usize,
                                    jw1 as usize..jw2 as usize
                                ]);
                                if img.shape() != w_.shape() {
                                    return Err("Shape unexpected".into());
                                }
                                let copied_img = ndarray::Array::from_shape_vec(
                                    img.raw_dim(),
                                    img.iter().cloned().collect(),
                                )?;
                                let imgs = copied_img
                                    .view()
                                    .into_shape((1, img.shape().iter().product()))?;
                                let copied_w = ndarray::Array::from_shape_vec(
                                    w_.raw_dim(),
                                    w_.iter().cloned().collect(),
                                )?;
                                let w_s = copied_w
                                    .view()
                                    .into_shape((w_.shape().iter().product(), 1))?;
                                ndarray::ArrayView2::dot(&imgs, &w_s)[[0, 0]]
                            } else {
                                let copied_img = ndarray::Array::from_shape_vec(
                                    img.raw_dim(),
                                    img.iter().cloned().collect(),
                                )?;
                                let imgs = copied_img
                                    .view()
                                    .into_shape((1, img.shape().iter().product()))?;
                                let copied_w = ndarray::Array::from_shape_vec(
                                    w.raw_dim(),
                                    w.iter().cloned().collect(),
                                )?;
                                let ws = copied_w
                                    .view()
                                    .into_shape((w.shape().iter().product(), 1))?;
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
) -> Result<ArrayType, Box<dyn std::error::Error>> {
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

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_conv.py
/// https://onnx.ai/onnx/operators/onnx__Conv.html
pub fn conv(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64, // defined but never used because even thought Conv has 2 versions they both do the same thing
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    match inputs.len() {
        2 => {
            let attributes = ConvAttributes::new(node, inputs[0], inputs[1]);
            conv_impl(inputs[0], inputs[1], None, attributes)
        }
        3 => {
            let attributes = ConvAttributes::new(node, inputs[0], inputs[1]);
            conv_impl(inputs[0], inputs[1], Some(inputs[2]), attributes)
        }
        _ => {
            panic!("Unexpected error: convolution has {} inputs", inputs.len());
        }
    }
}
