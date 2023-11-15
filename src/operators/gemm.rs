#![allow(unused_variables, dead_code)]
use ndarray::{ArrayD, Ix2};
use trait_set::trait_set;

// TODO: remove this when operator is implemented
use crate::{
    onnx::NodeProto,
    utils::{pick_opset_version, ArrayType},
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

trait_set! {
    pub trait ArrayNumericValueTrait<A: std::ops::Mul> = Clone
    + Copy
    + num::Zero
    + std::ops::Mul<Output = A>
    + std::ops::Add
    + std::ops::AddAssign<<A as std::ops::Mul>::Output>
    + std::ops::MulAssign<<A as std::ops::Mul>::Output>
}

//// Compute Y = alpha * A’ * B’ + beta * C, 
/// where input tensor A has shape (M, K) or (K, M), 
/// input tensor B has shape (K, N) or (N, K), 
/// input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N). 
/// A will be transposed before doing the computation if attribute transA is non-zero, same for B and transB.
fn dot_product<A: ArrayNumericValueTrait<A>>(
    lhs: &ArrayD<A>,
    rhs: &ArrayD<A>,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let lhs_shape_len = lhs_shape.len();
    let rhs_shape_len = rhs_shape.len();

    // if lhs_shape_len == 1 && rhs_shape_len == 1 {
    //     // If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
    //     let alen = lhs.shape()[0];
    //     let blen = rhs.shape()[0];
    //     if alen != blen {
    //         return Err("Gemm: a and b must have the same length".into());
    //     }
    //     let mut res = A::zero();
    //     for i in 0..lhs.shape()[0] {
    //         res += lhs[i] * rhs[i];
    //     }
    //     return Ok(ArrayD::from_elem(ndarray::IxDyn(&[]), res));
    // } else 
    if lhs_shape_len == 2 && rhs_shape_len == 2 {
        // If both a and b are 2-D arrays, it is matrix multiplication
        let lhs = lhs.view().into_dimensionality::<Ix2>()?;
        let rhs = rhs.view().into_dimensionality::<Ix2>()?;
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        if lhs_shape[1] != rhs_shape[0] {
            return Err("Gemm: a and b must have compatible shapes".into());
        }
        let mut res = ArrayD::zeros(ndarray::IxDyn(&[lhs_shape[0], rhs_shape[1]]));
        for i in 0..lhs_shape[0] {
            for j in 0..rhs_shape[1] {
                for k in 0..lhs_shape[1] {
                    res[[i, j]] += lhs[[i, k]] * rhs[[k, j]];
                }
            }
        }
        return Ok(res);
    } else {
        return Err("Gemm: a and b must be 2D matrices".into());
    }
    // else if lhs_shape_len == 0 || rhs_shape_len == 0 {
    //     // If either a or b is 0-D (scalar), it is equivalent to multiply
    //     if lhs_shape_len == 0 {
    //         // lhs is scalar
    //         let mut res = rhs.clone();
    //         let scalar_val = lhs.sum(); // FIXME: find a better way to get scalar value
    //         for i in res.iter_mut() {
    //             *i *= scalar_val;
    //         }
    //         return Ok(res);
    //     } else {
    //         // rhs is scalar
    //         let mut res = lhs.clone();
    //         let scalar_val = rhs.sum(); // FIXME: find a better way to get scalar value
    //         for i in res.iter_mut() {
    //             *i *= scalar_val;
    //         }
    //         return Ok(res);
    //     }
    // } else if rhs_shape_len == 1 {
    //     // If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
    //     let lhs_last_len = lhs_shape[lhs_shape_len - 1];
    //     let rhs_len = rhs_shape[0];
    //     if lhs_last_len != rhs_len {
    //         return Err("Gemm: a and b must have the same length".into());
    //     }
    //     let mut res = ArrayD::<A>::zeros(ndarray::IxDyn(&lhs_shape[..lhs_shape_len - 1]));
    //     // TODO
    //     return Ok(res);
    // } else {
    //     // If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:
    //     let lhs_last_len = lhs_shape[lhs_shape_len - 1];
    //     let rhs_second_last_len = rhs_shape[rhs_shape_len - 2];
    //     if lhs_last_len != rhs_second_last_len {
    //         return Err("Gemm: a and b must have aligned shapes".into());
    //     }
    //     let new_shape = lhs_shape[..lhs_shape_len - 1]
    //         .iter()
    //         .copied()
    //         .chain(
    //             rhs_shape
    //                 .iter()
    //                 .enumerate()
    //                 .skip_while(|(i, x)| *i == rhs_shape_len - 2)
    //                 .map(|(i, x)| *x),
    //         )
    //         .collect::<Vec<_>>();
    //     let mut res = ArrayD::<A>::zeros(ndarray::IxDyn(&new_shape));
    //     return Ok(res);
    // }
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

fn _gem00<
    A: ArrayNumericValueTrait<A>,
>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    alpha: f32,
    beta: f32,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
    let o = dot_product(a, b)? * alpha;
    if let Some(c) = c {
        if beta != 0.0 {
            Ok(o + c * beta)
        } else {
            Ok(o)
        }
    } else {
        Ok(o)
    }
}

fn _gem01<
    A: ArrayNumericValueTrait<A>,
>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    alpha: f32,
    beta: f32,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {

}

fn _gem10<
    A: ArrayNumericValueTrait<A>,
>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    alpha: f32,
    beta: f32,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
}

fn _gem11<
    A: ArrayNumericValueTrait<A>,
>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    alpha: f32,
    beta: f32,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
}

fn gemm_6<
    A: ArrayNumericValueTrait<A>,
>(
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
    if attrs.broadcast {
        let res = _meth(a, b, None, attrs.alpha, attrs.beta)?;
        if let Some(c) = c {
            if c.shape() != res.shape() {
                Err("Gemm: c shape does not match result shape".into())
            } else {
                Ok(res + c)
            }
        } else {
            Ok(res)
        }
    } else {
        _meth(a, b, c, attrs.alpha, attrs.beta)
    }
}

fn gemm_7<
    A: ArrayNumericValueTrait<A>,
>(
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

fn _gemm_internal<
    A: ArrayNumericValueTrait<A>,
>(
    a: &ArrayD<A>,
    b: &ArrayD<A>,
    c: Option<&ArrayD<A>>,
    attrs: GemmAttrs,
    target_version: i64,
) -> Result<ArrayD<A>, Box<dyn std::error::Error>> {
    let methver = if target_version >= 7 {
        gemm_7::<A>
    } else {
        gemm_6::<A>
    };
    methver(a, b, c, attrs)
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
        (Some(ArrayType::F32(a)), Some(ArrayType::F32(b)), Some(ArrayType::F32(c))) => Ok(
            ArrayType::F32(_gemm_internal(a, b, Some(c), attrs, target_version)?),
        ),
        (Some(ArrayType::F32(a)), Some(ArrayType::F32(b)), None) => Ok(ArrayType::F32(
            _gemm_internal(a, b, None, attrs, target_version)?,
        )),
        (Some(ArrayType::I64(a)), Some(ArrayType::I64(b)), Some(ArrayType::I64(c))) => Ok(
            ArrayType::I64(_gemm_internal(a, b, Some(c), attrs, target_version)?),
        ),
        (Some(ArrayType::I64(a)), Some(ArrayType::I64(b)), None) => Ok(ArrayType::I64(
            _gemm_internal(a, b, None, attrs, target_version)?,
        )),
        _ => todo!("Gemm for type {:?}", inputs),
    }
}
