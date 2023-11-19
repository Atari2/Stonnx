use crate::{
    onnx::NodeProto,
    utils::{pick_opset_version, ArrayType, BoxResult, OperationResult},
};
use ndarray::{ArrayD, Ix0};
use rand::{Rng, SeedableRng};
const OPSET_VERSIONS: [i64; 6] = [1, 6, 7, 10, 12, 13];

#[derive(Debug)]
struct DropoutAttrs {
    is_test: bool,
    ratio: f32,
    seed: i64,
}

// attrs_1 | attrs_6   => is_test: int (default 0)
//                        ratio: float (default 0.5)
// attrs_7 | attrs_10  => ratio: float (default 0.5)
// attrs_12 | attrs_13 => seed: int (default 0)
impl DropoutAttrs {
    fn new(node: &NodeProto, version: i64) -> Self {
        if version < 6 {
            // is_test, ratio
            Self {
                is_test: node
                    .attribute
                    .iter()
                    .find(|a| a.name() == "is_test")
                    .map_or(false, |a| a.i.unwrap_or(0) != 0),
                ratio: node
                    .attribute
                    .iter()
                    .find(|a| a.name() == "ratio")
                    .map_or(0.5, |a| a.f.unwrap_or(0.5)),
                seed: 0,
            }
        } else if version > 6 && version < 12 {
            Self {
                is_test: false,
                ratio: node
                    .attribute
                    .iter()
                    .find(|a| a.name() == "ratio")
                    .map_or(0.5, |a| a.f.unwrap_or(0.5)),
                seed: 0,
            }
        } else {
            let seed: i64 = rand::thread_rng().gen();
            Self {
                is_test: false,
                ratio: 0.5,
                seed: node
                    .attribute
                    .iter()
                    .find(|a| a.name() == "seed")
                    .map_or(seed, |a| a.i.unwrap_or(seed)),
            }
        }
    }
}

fn dropout_common_f32(
    data: &ArrayD<f32>,
    training_mode: bool,
    attrs: DropoutAttrs,
    output_len: usize,
) -> BoxResult<(ArrayType, Option<ArrayType>)> {
    let return_mask = output_len == 2;
    if attrs.ratio == 0.0 || !training_mode {
        if !return_mask {
            Ok((ArrayType::F32(data.clone()), None))
        } else {
            Ok((
                ArrayType::F32(data.clone()),
                Some(ArrayType::Bool(ArrayD::from_elem(data.shape(), true))),
            ))
        }
    } else {
        let mut rng = rand::rngs::StdRng::seed_from_u64(attrs.seed as u64);
        let distribution = rand::distributions::Uniform::new(0.0, 1.0);
        let mask = ArrayD::from_shape_simple_fn(data.shape(), || rng.sample(distribution))
            .mapv(|x| x >= attrs.ratio);
        let scale = 1.0 / (1.0 - attrs.ratio);
        let btf = |a| if a { 1.0 } else { 0.0 };
        if !return_mask {
            Ok((ArrayType::F32(mask.mapv(btf) * data * scale), None))
        } else {
            Ok((
                ArrayType::F32(mask.mapv(btf) * data * scale),
                Some(ArrayType::Bool(mask)),
            ))
        }
    }
}

fn dropout_common(
    data: &ArrayType,
    training_mode: bool,
    attrs: DropoutAttrs,
    output_len: usize,
) -> BoxResult<(ArrayType, Option<ArrayType>)> {
    match data {
        ArrayType::F32(data) => dropout_common_f32(data, training_mode, attrs, output_len),
        _ => todo!("Dropout for type {}", data),
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_dropout.py
/// https://onnx.ai/onnx/operators/onnx__Dropout.html
pub fn dropout(
    inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
    output_len: usize,
) -> BoxResult<OperationResult> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    let mut attrs = DropoutAttrs::new(node, target_version);
    if attrs.is_test && target_version < 7 {
        return Ok((inputs[0].clone(), None).into());
    }
    if target_version < 12 {
        Ok(dropout_common(inputs[0], false, attrs, output_len)?.into())
    } else {
        attrs.ratio = if let Some(ratio) = inputs.get(1) {
            match ratio {
                ArrayType::F32(ratio) => ratio.clone().into_dimensionality::<Ix0>()?.into_scalar(),
                _ => return Err("Ratio must be a scalar".into()),
            }
        } else {
            0.5
        };
        let training_mode = if let Some(mode) = inputs.get(2) {
            match mode {
                ArrayType::I64(mode) => {
                    mode.clone().into_dimensionality::<Ix0>()?.into_scalar() != 0
                }
                _ => return Err("Training mode must be a scalar".into()),
            }
        } else {
            false
        };
        Ok(dropout_common(inputs[0], training_mode, attrs, output_len)?.into())
    }
}