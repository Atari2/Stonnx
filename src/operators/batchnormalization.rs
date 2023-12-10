use ndarray::{ArrayD, SliceInfoElem};

use crate::common::{ArrayElement, BoxResult, F32IntoType, OperatorResult, TensorType};
use crate::onnx::NodeProto;
use crate::utils::pick_opset_version;

const OPSET_VERSIONS: [i64; 6] = [1, 6, 7, 9, 14, 15];

/// ver 1 | ver 6
/// inputs: X, scale, B, mean , var
/// outputs: Y, mean (opt), var(opt), saved_mean(opt), saved_var(opt)
/// attributes: epsilon(default 1e-5), is_test(default 0), momentum (default 0.9), spatial(default 1)

/// ver 7
/// inputs: X, scale, B, mean , var
/// outputs: Y, mean (opt), var(opt), saved_mean(opt), saved_var(opt)
/// attributes: epsilon(default 1e-5), momentum (default 0.9), spatial(default 1)

/// ver 9
/// inputs: X, scale, B, mean , var
/// outputs: Y, mean (opt), var(opt), saved_mean(opt), saved_var(opt)
/// attributes: epsilon(default 1e-5), momentum (default 0.9)

/// ver 14 | 15
/// inputs: X, scale, B, input_mean, input_var
/// outputs: Y, running_mean (opt), running_var(opt)
/// attributes: epsilon(default 1e-5), momentum (default 0.9), training_mode(default 0)

#[derive(Debug)]
struct BatchNormalizationAttrs {
    epsilon: f32,
    momentum: Option<f32>,
    _spatial: bool, // unused?
    is_test: bool,
    training_mode: bool,
}

impl BatchNormalizationAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            epsilon: node
                .attribute
                .iter()
                .find(|a| a.name() == "epsilon")
                .map_or(1e-5, |a| a.f.unwrap_or(1e-5)),
            momentum: node
                .attribute
                .iter()
                .find(|a| a.name() == "momentum")
                .map(|a| a.f.unwrap_or(0.9)),
            _spatial: node
                .attribute
                .iter()
                .find(|a| a.name() == "spatial")
                .map_or(true, |a| a.i.unwrap_or(1) == 1),
            is_test: node
                .attribute
                .iter()
                .find(|a| a.name() == "is_test")
                .map_or(false, |a| a.i.unwrap_or(0) == 1),
            training_mode: node
                .attribute
                .iter()
                .find(|a| a.name() == "training_mode")
                .map_or(false, |a| a.i.unwrap_or(0) == 1),
        }
    }
}

fn _batchnorm_test_mode<A: ArrayElement>(
    x: &ArrayD<A>,
    scale: &ArrayD<A>,
    bias: &ArrayD<A>,
    mean: &ArrayD<A>,
    var: &ArrayD<A>,
    epsilon: f32,
) -> BoxResult<TensorType>
where
    TensorType: From<ArrayD<A>>,
    f32: F32IntoType<A>,
{
    let dims_x = x.ndim();
    let dim_ones_generator = std::iter::repeat(1).take(dims_x - 2);
    let sshape = [scale.len()]
        .into_iter()
        .chain(dim_ones_generator.clone())
        .collect::<Vec<_>>();
    let s = scale.to_shape(sshape.as_slice())?;
    let bshape = [bias.len()]
        .into_iter()
        .chain(dim_ones_generator.clone())
        .collect::<Vec<_>>();
    let b = bias.to_shape(bshape.as_slice())?;
    let mshape = [mean.len()]
        .into_iter()
        .chain(dim_ones_generator.clone())
        .collect::<Vec<_>>();
    let m = mean.to_shape(mshape.as_slice())?;
    let vshape = [var.len()]
        .into_iter()
        .chain(dim_ones_generator.clone())
        .collect::<Vec<_>>();
    let v = var.to_shape(vshape.as_slice())?;
    let y = &s * (x - &m) / (v.mapv(|x| x + epsilon.as_())).mapv(|x| x.sqrt()) + b;
    Ok(y.into())
}

// FIXME: remove this num::Float requirement, it is needed for ArrayBase::var
fn _batchnorm_training_mode<A: ArrayElement + num::Float>(
    x: &ArrayD<A>,
    scale: &ArrayD<A>,
    bias: &ArrayD<A>,
    mean: &ArrayD<A>,
    var: &ArrayD<A>,
    momentum: f32,
    epsilon: f32,
) -> BoxResult<Vec<TensorType>>
where
    TensorType: From<ArrayD<A>>,
    f32: F32IntoType<A>,
{
    let momentum = momentum.as_();
    let axis = (0..x.ndim()).skip_while(|&i| i == 1).collect::<Vec<_>>();
    let mut saved_mean = x.clone();
    for ax in axis.iter().rev() {
        saved_mean = saved_mean
            .mean_axis(ndarray::Axis(*ax))
            .ok_or(anyhow::anyhow!("BatchNormalization: mean_axis failed"))?;
    }
    let saved_var_len = x.shape()[1];
    let mut saved_var = ArrayD::<A>::zeros(vec![saved_var_len]);
    for i in 0..saved_var_len {
        let sliceinfo = [(0..).into(), i.into()]
            .into_iter()
            .chain(std::iter::repeat((0..).into()).take(x.ndim() - 2))
            .collect::<Vec<SliceInfoElem>>();
        let sliced = x.slice(sliceinfo.as_slice()).to_owned();
        saved_var[i] = sliced.var((0.0).as_());
    }
    let output_mean =
        mean.mapv(|x| x * momentum) + saved_mean.mapv(|x| x * (1.0f32.as_() - momentum));
    let output_var = var.mapv(|x| x * momentum) + saved_var.mapv(|x| x * (1.0f32.as_() - momentum));
    let y = _batchnorm_test_mode(x, scale, bias, &output_mean, &output_var, epsilon)?;
    Ok(vec![
        y,
        saved_mean.into(),
        saved_var.into(),
        output_mean.into(),
        output_var.into(),
    ])
}

fn batchnormalization_1_6<A: ArrayElement + num::Float>(
    x: &ArrayD<A>,
    scale: &ArrayD<A>,
    bias: &ArrayD<A>,
    mean: &ArrayD<A>,
    var: &ArrayD<A>,
    attrs: BatchNormalizationAttrs,
) -> BoxResult<Vec<TensorType>>
where
    TensorType: From<ArrayD<A>>,
    f32: F32IntoType<A>,
{
    if attrs.is_test {
        Ok(vec![_batchnorm_test_mode(
            x,
            scale,
            bias,
            mean,
            var,
            attrs.epsilon,
        )?])
    } else {
        _batchnorm_training_mode(
            x,
            scale,
            bias,
            mean,
            var,
            attrs.momentum.unwrap_or(0.9),
            attrs.epsilon,
        )
    }
}
fn batchnormalization_7_9<A: ArrayElement + num::Float>(
    x: &ArrayD<A>,
    scale: &ArrayD<A>,
    bias: &ArrayD<A>,
    mean: &ArrayD<A>,
    var: &ArrayD<A>,
    attrs: BatchNormalizationAttrs,
) -> BoxResult<Vec<TensorType>>
where
    TensorType: From<ArrayD<A>>,
    f32: F32IntoType<A>,
{
    if let Some(momentum) = attrs.momentum {
        let momentum = momentum.as_();
        let axis = (0..x.ndim()).filter(|i| *i != 1).collect::<Vec<_>>();
        let mut saved_mean = x.clone();
        for ax in axis.iter().rev() {
            saved_mean = saved_mean
                .mean_axis(ndarray::Axis(*ax))
                .ok_or(anyhow::anyhow!("BatchNormalization: mean_axis failed"))?;
        }
        let saved_var_len = x.shape()[1];
        let mut saved_var = ArrayD::<A>::zeros(vec![saved_var_len]);
        for i in 0..saved_var_len {
            let sliceinfo = [(0..).into(), i.into()]
                .into_iter()
                .chain(std::iter::repeat((0..).into()).take(x.ndim() - 2))
                .collect::<Vec<SliceInfoElem>>();
            let sliced = x.slice(sliceinfo.as_slice()).to_owned();
            saved_var[i] = sliced.var((0.0).as_());
        }
        let output_mean =
            mean.mapv(|x| x * momentum) + saved_mean.mapv(|x| x * (1f32.as_() - momentum));
        let output_var =
            var.mapv(|x| x * momentum) + saved_var.mapv(|x| x * (1f32.as_() - momentum));
        let y = _batchnorm_test_mode(x, scale, bias, &output_mean, &output_var, attrs.epsilon)?;
        Ok(vec![y])
    } else {
        Ok(vec![_batchnorm_test_mode(
            x,
            scale,
            bias,
            mean,
            var,
            attrs.epsilon,
        )?])
    }
}
fn batchnormalization_14_15<A: ArrayElement + num::Float>(
    x: &ArrayD<A>,
    scale: &ArrayD<A>,
    bias: &ArrayD<A>,
    mean: &ArrayD<A>,
    var: &ArrayD<A>,
    attrs: BatchNormalizationAttrs,
) -> BoxResult<Vec<TensorType>>
where
    TensorType: From<ArrayD<A>>,
    f32: F32IntoType<A>,
{
    if !attrs.training_mode {
        let res = _batchnorm_test_mode(x, scale, bias, mean, var, attrs.epsilon)?;
        Ok(vec![res])
    } else {
        let outputs = _batchnorm_training_mode(
            x,
            scale,
            bias,
            mean,
            var,
            attrs.momentum.unwrap_or(0.9),
            attrs.epsilon,
        )?;
        Ok(outputs
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| if i == 1 || i == 2 { None } else { Some(v) })
            .collect::<Vec<_>>())
    }
}

/// Carries out batch normalization as described in the paper <https://arxiv.org/abs/1502.03167>.
///
/// Depending on the mode it is being run, There are five required inputs ‘X’, ‘scale’, ‘B’, ‘input_mean’ and ‘input_var’.
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_batch_normalization.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html)
pub fn batchnormalization(
    inputs: &[&TensorType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let attrs = BatchNormalizationAttrs::new(node);
    let target_ver = pick_opset_version(opset_version, &OPSET_VERSIONS);
    if inputs.len() < 5 {
        return Err(anyhow::anyhow!(
            "BatchNormalization: inputs must be at least 5"
        ));
    }
    let x = if let TensorType::F32(x) = inputs[0] {
        x
    } else {
        todo!("BatchNormalization for x type {}", inputs[0])
    };
    let scale = if let TensorType::F32(scale) = inputs[1] {
        scale
    } else {
        todo!("BatchNormalization for scale type {}", inputs[1])
    };
    let bias = if let TensorType::F32(bias) = inputs[2] {
        bias
    } else {
        todo!("BatchNormalization for bias type {}", inputs[2])
    };
    let mean = if let TensorType::F32(mean) = inputs[3] {
        mean
    } else {
        todo!("BatchNormalization for mean type {}", inputs[3])
    };
    let var = if let TensorType::F32(var) = inputs[4] {
        var
    } else {
        todo!("BatchNormalization for var type {}", inputs[4])
    };

    if target_ver < 7 {
        Ok(batchnormalization_1_6(x, scale, bias, mean, var, attrs)?.into())
    } else if target_ver < 14 {
        Ok(batchnormalization_7_9(x, scale, bias, mean, var, attrs)?.into())
    } else {
        Ok(batchnormalization_14_15(x, scale, bias, mean, var, attrs)?.into())
    }
}
