#![allow(unused_variables, dead_code)]
use ndarray::{ArrayD, ArrayViewD, Ix2};
use trait_set::trait_set;

use crate::{
    common::{BoxResult, OperatorResult, TensorType},
    onnx::NodeProto,
    utils::pick_opset_version,
};
use anyhow::anyhow;

const OPSET_VERSIONS: [i64; 6] = [1, 6, 7, 9, 11, 13];

#[derive(Debug)]
struct GemmAttrs {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    broadcast: bool,
}

trait_set! {
    pub trait ArrayNumericValueTrait<A: std::ops::Mul> = Clone
    + Copy
    + num::Zero
    + std::ops::Mul<Output = A>
    + std::ops::Add
    + std::ops::AddAssign<<A as std::ops::Mul>::Output>
}

/// Compute Y = alpha * A’ * B’ + beta * C,
///
/// where input tensor A has shape (M, K) or (K, M),
///
/// input tensor B has shape (K, N) or (N, K),
///
/// input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N).
///
/// A will be transposed before doing the computation if attribute transA is non-zero, same for B and transB.
fn dot_product<'a, A: ArrayNumericValueTrait<A>>(
    lhs: ArrayViewD<'a, A>,
    rhs: ArrayViewD<'a, A>,
) -> BoxResult<ArrayD<A>> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let lhs_shape_len = lhs_shape.len();
    let rhs_shape_len = rhs_shape.len();
    if lhs_shape_len == 2 && rhs_shape_len == 2 {
        // If both a and b are 2-D arrays, it is matrix multiplication
        let lhs = lhs.view().into_dimensionality::<Ix2>()?;
        let rhs = rhs.view().into_dimensionality::<Ix2>()?;
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        if lhs_shape[1] != rhs_shape[0] {
            return Err(anyhow!("Gemm: a and b must have compatible shapes"));
        }
        let mut res = ArrayD::zeros(ndarray::IxDyn(&[lhs_shape[0], rhs_shape[1]]));
        for i in 0..lhs_shape[0] {
            for j in 0..rhs_shape[1] {
                for k in 0..lhs_shape[1] {
                    res[[i, j]] += lhs[[i, k]] * rhs[[k, j]];
                }
            }
        }
        Ok(res)
    } else {
        Err(anyhow!("Gemm: a and b must be 2D matrices"))
    }
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
                .find(|a| a.name() == "transA")
                .map_or(false, |a| a.i.unwrap_or(0) == 1),
            trans_b: node
                .attribute
                .iter()
                .find(|a| a.name() == "transB")
                .map_or(false, |a| a.i.unwrap_or(0) == 1),
            broadcast: node
                .attribute
                .iter()
                .find(|a| a.name() == "broadcast")
                .map_or(false, |a| a.i.unwrap_or(0) == 1),
        }
    }
}

fn _gemm_common_f32(
    a: &ArrayD<f32>,
    b: &ArrayD<f32>,
    c: Option<&ArrayD<f32>>,
    attrs: GemmAttrs,
) -> BoxResult<ArrayD<f32>> {
    let at = if attrs.trans_a { a.t() } else { a.view() };
    let bt = if attrs.trans_b { b.t() } else { b.view() };
    let o = dot_product(at, bt)? * attrs.alpha;
    if let Some(c) = c {
        if attrs.beta != 0.0 {
            Ok(o + c * attrs.beta)
        } else {
            Ok(o)
        }
    } else {
        Ok(o)
    }
}

fn _gemm_common_i64(
    a: &ArrayD<i64>,
    b: &ArrayD<i64>,
    c: Option<&ArrayD<i64>>,
    attrs: GemmAttrs,
) -> BoxResult<ArrayD<i64>> {
    let at = if attrs.trans_a { a.t() } else { a.view() };
    let bt = if attrs.trans_b { b.t() } else { b.view() };
    let o = dot_product(at, bt)?;
    let oc = o.mapv(|v| v as f32) * attrs.alpha;
    if let Some(c) = c {
        let cc = c.mapv(|v| v as f32);
        if attrs.beta != 0.0 {
            let res = oc + cc * attrs.beta;
            Ok(res.mapv(|v| v as i64))
        } else {
            Ok(o)
        }
    } else {
        Ok(o)
    }
}

fn gemm_6(
    a: &TensorType,
    b: &TensorType,
    c: Option<&TensorType>,
    attrs: GemmAttrs,
) -> BoxResult<TensorType> {
    match (a, b, c) {
        (TensorType::F32(a), TensorType::F32(b), Some(TensorType::F32(c))) => {
            if !attrs.broadcast {
                let res = _gemm_common_f32(a, b, Some(c), attrs)?;
                if c.shape() != res.shape() {
                    return Err(anyhow!("Gemm: c and res must have the same shape"));
                }
                Ok(TensorType::F32(res + c))
            } else {
                Ok(TensorType::F32(_gemm_common_f32(a, b, Some(c), attrs)?))
            }
        }
        (TensorType::F32(a), TensorType::F32(b), None) => {
            Ok(TensorType::F32(_gemm_common_f32(a, b, None, attrs)?))
        }
        (TensorType::I64(a), TensorType::I64(b), Some(TensorType::I64(c))) => {
            if !attrs.broadcast {
                let res = _gemm_common_i64(a, b, Some(c), attrs)?;
                if c.shape() != res.shape() {
                    return Err(anyhow!("Gemm: c and res must have the same shape"));
                }
                Ok(TensorType::I64(res + c))
            } else {
                Ok(TensorType::I64(_gemm_common_i64(a, b, Some(c), attrs)?))
            }
        }
        (TensorType::I64(a), TensorType::I64(b), None) => {
            Ok(TensorType::I64(_gemm_common_i64(a, b, None, attrs)?))
        }
        _ => {
            todo!("GEMM: {} {} {:?}", a, b, c)
        }
    }
}

fn gemm_7(
    a: &TensorType,
    b: &TensorType,
    c: Option<&TensorType>,
    attrs: GemmAttrs,
) -> BoxResult<TensorType> {
    match (a, b, c) {
        (TensorType::F32(a), TensorType::F32(b), Some(TensorType::F32(c))) => {
            Ok(TensorType::F32(_gemm_common_f32(a, b, Some(c), attrs)?))
        }
        (TensorType::F32(a), TensorType::F32(b), None) => {
            Ok(TensorType::F32(_gemm_common_f32(a, b, None, attrs)?))
        }
        (TensorType::I64(a), TensorType::I64(b), Some(TensorType::I64(c))) => {
            Ok(TensorType::I64(_gemm_common_i64(a, b, Some(c), attrs)?))
        }
        (TensorType::I64(a), TensorType::I64(b), None) => {
            Ok(TensorType::I64(_gemm_common_i64(a, b, None, attrs)?))
        }
        (a, b, c) => {
            todo!("GEMM: {:?} {:?} {:?}", a, b, c)
        }
    }
}

fn _gemm_internal(
    a: &TensorType,
    b: &TensorType,
    c: Option<&TensorType>,
    attrs: GemmAttrs,
    target_version: i64,
) -> BoxResult<TensorType> {
    if target_version >= 7 {
        gemm_7(a, b, c, attrs)
    } else {
        gemm_6(a, b, c, attrs)
    }
}

/// General Matrix multiplication: <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3>
///
/// - A’ = transpose(A) if transA else A
/// - B’ = transpose(B) if transB else B
///
/// Compute Y = alpha * A’ * B’ + beta * C, where input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N).
///
/// A will be transposed before doing the computation if attribute transA is non-zero, same for B and transB.
///
/// [Python reference](<https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_gemm.py>)
///
/// [ONNX Documentation](<https://onnx.ai/onnx/operators/onnx__Gemm.html>)
pub fn gemm(
    inputs: &[&TensorType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    let attrs = GemmAttrs::new(node);

    match (inputs.get(0), inputs.get(1), inputs.get(2)) {
        (Some(a), Some(b), Some(c)) => {
            Ok(_gemm_internal(a, b, Some(c), attrs, target_version)?.into())
        }
        (Some(a), Some(b), None) => Ok(_gemm_internal(a, b, None, attrs, target_version)?.into()),
        _ => Err(anyhow!("Gemm: invalid inputs")),
    }
}
