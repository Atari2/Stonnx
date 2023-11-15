#![allow(unused_variables, dead_code)]
use ndarray::ArrayD;

// TODO: remove this when operator is implemented
use crate::{
    onnx::NodeProto,
    utils::{pick_opset_version, ArrayType, ArrayTypeTrait},
};

const OPSET_VERSIONS: [i64; 6] = [1, 6, 7, 9, 11, 13];

#[derive(Debug)]
struct GemmAttrs {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    broadcast: bool,
}

impl GemmAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            alpha: node
                .attribute
                .iter()
                .find(|a| a.name() == "alpha")
                .map_or(1.0, |a| a.f.unwrap_or(1.0)),
            beta: node
                .attribute
                .iter()
                .find(|a| a.name() == "beta")
                .map_or(1.0, |a| a.f.unwrap_or(1.0)),
            trans_a: node
                .attribute
                .iter()
                .find(|a| a.name() == "trans_a")
                .map_or(false, |a| a.i.unwrap_or(0) == 1),
            trans_b: node
                .attribute
                .iter()
                .find(|a| a.name() == "trans_b")
                .map_or(false, |a| a.i.unwrap_or(0) == 1),
            broadcast: node
                .attribute
                .iter()
                .find(|a| a.name() == "broadcast")
                .map_or(false, |a| a.i.unwrap_or(0) == 1),
        }
    }
}

fn _gem00<A: Clone + Copy + num::Zero>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    alpha: f32,
    beta: f32,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
}

fn _gem01<A: Clone + Copy + num::Zero>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    alpha: f32,
    beta: f32,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
}

fn _gem10<A: Clone + Copy + num::Zero>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    alpha: f32,
    beta: f32,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
}

fn _gem11<A: Clone + Copy + num::Zero>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    alpha: f32,
    beta: f32,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
}

fn gemm_6<A: Clone + Copy + num::Zero>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    attrs: GemmAttrs,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
    if attrs.broadcast {
        let _meth = if attrs.trans_a {
            if attrs.trans_b {
                _gem11::<A>
            } else {
                _gem10::<A>
            }
        } else if attrs.trans_b {
            _gem01::<A>
        } else {
            _gem00::<A>
        };
        let res = _meth(a, b, c, attrs.alpha, attrs.beta)?;
        if let Some(c) = c {
            if c.shape() != res.shape() {
                Err("Gemm: c shape does not match result shape".into())
            } else {
                Ok(res + c)
            }
        } else {
            Ok(res)
        }
    }
}

fn gemm_7<A: Clone + Copy + num::Zero>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    attrs: GemmAttrs,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
    let _meth = if attrs.trans_a {
        if attrs.trans_b {
            _gem11::<A>
        } else {
            _gem10::<A>
        }
    } else if attrs.trans_b {
        _gem01::<A>
    } else {
        _gem00::<A>
    };
    _meth(a, b, c, attrs.alpha, attrs.beta)
}

fn _gemm_internal<A: Clone + Copy + num::Zero>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    attrs: GemmAttrs,
    target_version: i64,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
    let methver = if target_version >= 7 { gemm_7 } else { gemm_6 };
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_gemm.py
/// https://onnx.ai/onnx/operators/onnx__Gemm.html
pub fn gemm(
    inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    let attrs = GemmAttrs::new(node);

    match (inputs.get(0), inputs.get(1), inputs.get(2)) {
        (Some(ArrayType::F32(a)), Some(ArrayType::F32(b)), Some(ArrayType::F32(c))) => {
            Ok(ArrayType::F32(_gemm_internal(a, b, Some(c), attrs, target_version)?))
        },
        (Some(ArrayType::F32(a)), Some(ArrayType::F32(b)), None) => {
            Ok(ArrayType::F32(_gemm_internal(a, b, None, attrs, target_version)?))
        },
        (Some(ArrayType::I64(a)), Some(ArrayType::I64(b)), Some(ArrayType::I64(c))) => {
            Ok(ArrayType::I64(_gemm_internal(a, b, Some(c), attrs, target_version)?))
        },
        (Some(ArrayType::I64(a)), Some(ArrayType::I64(b)), None) => {
            Ok(ArrayType::I64(_gemm_internal(a, b, None, attrs, target_version)?))
        },
        _ => todo!("Gemm for type {:?}", inputs),
    }
}
