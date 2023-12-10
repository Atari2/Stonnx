use crate::{
    common::{BoxResult, OperatorResult, TensorType},
    onnx::NodeProto,
    utils::pick_opset_version,
};
use anyhow::anyhow;

const OPSET_VERSIONS: [i64; 5] = [1, 6, 11, 12, 13];

#[derive(Debug)]
struct ClipAttrs {
    max: f32,
    min: f32,
}

impl ClipAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            max: node
                .attribute
                .iter()
                .find(|a| a.name() == "max")
                .map_or(std::f32::MAX, |a| a.f.unwrap_or(std::f32::MAX)),
            min: node
                .attribute
                .iter()
                .find(|a| a.name() == "min")
                .map_or(std::f32::MIN, |a| a.f.unwrap_or(std::f32::MIN)),
        }
    }
}

fn clip_6(inputs: &[&TensorType], attrs: ClipAttrs) -> BoxResult<TensorType> {
    if let TensorType::F32(a) = &inputs[0] {
        let mut a = a.to_owned();
        a.mapv_inplace(|v| v.max(attrs.min));
        a.mapv_inplace(|v| v.min(attrs.max));
        Ok(TensorType::F32(a))
    } else {
        todo!("Clip for type {}", inputs[0])
    }
}

fn clip_11(inputs: &[&TensorType]) -> BoxResult<TensorType> {
    let ilen = inputs.len();
    if ilen == 1 {
        return Ok(inputs[0].to_owned());
    }
    let amin = match inputs.get(1) {
        Some(a) => {
            if !a.shape().is_empty() {
                return Err(anyhow!("Amin must be a scalar"));
            } else if let TensorType::F32(a) = a {
                Some(a.sum())
            } else {
                todo!("Clip amin for type {}", a)
            }
        }
        None => None,
    };
    let amax = match inputs.get(2) {
        Some(a) => {
            if !a.shape().is_empty() {
                return Err(anyhow!("Amax must be a scalar"));
            } else if let TensorType::F32(a) = a {
                Some(a.sum())
            } else {
                todo!("Clip amax for type {}", a)
            }
        }
        None => None,
    };
    if let TensorType::F32(a) = &inputs[0] {
        let mut a = a.to_owned();
        if let Some(amin) = amin {
            a.mapv_inplace(|v| v.max(amin));
        }
        if let Some(amax) = amax {
            a.mapv_inplace(|v| v.min(amax));
        }
        Ok(TensorType::F32(a))
    } else {
        todo!("CLIP for type {}", inputs[0])
    }
}

/// Performs element-wise clipping of the input tensor.
///
/// Clip operator limits the given input within an interval. The interval is specified by the inputs 'min' and 'max'. The output tensor has the same shape and data type as the input tensor.
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_clip.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__Clip.html)
pub fn clip(
    inputs: &[&TensorType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    if target_version < 11 {
        // 1, 6
        Ok(clip_6(inputs, ClipAttrs::new(node))?.into())
    } else {
        // 11, 12, 13
        Ok(clip_11(inputs)?.into())
    }
}
