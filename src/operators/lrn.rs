use ndarray::{s, ArrayD, Axis};

use crate::{
    common::{BoxResult, OperatorResult, TensorType},
    onnx::NodeProto,
};
use anyhow::anyhow;

const _OPSET_VERSIONS: [i64; 2] = [1, 13];

#[derive(Debug)]
struct LRNAttrs {
    alpha: f32,
    beta: f32,
    bias: f32,
    size: i64,
}

impl LRNAttrs {
    fn new(node: &NodeProto) -> BoxResult<Self> {
        Ok(Self {
            alpha: node
                .attribute
                .iter()
                .find(|a| a.name() == "alpha")
                .map_or(0.0001, |a| a.f.unwrap_or(0.0001)),
            beta: node
                .attribute
                .iter()
                .find(|a| a.name() == "beta")
                .map_or(0.75, |a| a.f.unwrap_or(0.75)),
            bias: node
                .attribute
                .iter()
                .find(|a| a.name() == "bias")
                .map_or(1.0, |a| a.f.unwrap_or(1.0)),
            size: node
                .attribute
                .iter()
                .find(|a| a.name() == "size")
                .ok_or(anyhow!("size attribute is required"))?
                .i
                .ok_or(anyhow!("size attribute must be an integer"))?,
        })
    }
}

fn lrn_f32(input: &ArrayD<f32>, attrs: LRNAttrs) -> BoxResult<ArrayD<f32>> {
    if input.shape().len() != 4 {
        return Err(anyhow!("Input must be 4D"));
    }
    let mut square_sum = ArrayD::<f32>::zeros(input.shape());
    let minc = input.shape()[1] as isize;
    let c1 = (((attrs.size - 1) as f32) / 2f32).floor() as isize;
    let c2 = (((attrs.size - 1) as f32) / 2f32).ceil() as isize + 1;
    for c in 0..input.shape()[0] as isize {
        let begin = (c - c1).max(0);
        let end = (c + c2).min(minc);
        let mut sumpow = input.slice(s![.., begin..end, .., ..]).to_owned();
        sumpow.par_mapv_inplace(|v| v.powi(2));
        let sumpow = sumpow.sum_axis(Axis(1));
        square_sum.slice_mut(s![.., c, .., ..]).assign(&sumpow);
    }
    let mut biasarr = attrs.bias + (attrs.alpha / attrs.size as f32) * square_sum;
    biasarr.par_mapv_inplace(|v| v.powf(attrs.beta));
    let y = input / biasarr;
    Ok(y)
}

/// Local Response Normalization proposed in the AlexNet paper. It normalizes over local input regions.
///
/// The local region is defined across the channels.
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_lrn.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__LRN.html)
pub fn lrn(
    inputs: &[&TensorType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let attrs = LRNAttrs::new(node)?;
    match inputs.first() {
        Some(TensorType::F32(input)) => {
            let output = lrn_f32(input, attrs)?;
            Ok(TensorType::F32(output).into())
        }
        _ => todo!("LRN for type {:?}", inputs.first()),
    }
}
