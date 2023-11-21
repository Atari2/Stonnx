use ndarray::{ArrayD, ArrayViewD};
use num::{traits::AsPrimitive, Zero};
use protobuf::Enum;

use crate::{
    onnx::{tensor_proto::DataType, NodeProto},
    utils::{make_tensor_from_raw, ArrayType, BoxResult, NDIndex, OperationResult},
};

const _OPSET_VERSIONS: [i64; 3] = [1, 11, 13];

#[derive(Debug)]
struct GatherAttrs {
    axis: i64,
}

impl GatherAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            axis: node
                .attribute
                .iter()
                .find(|a| a.name() == "axis")
                .map_or(0, |a| a.i()),
        }
    }
}

fn _gather_generic<A: Clone + Copy + Zero, B: Clone + Zero + AsPrimitive<usize>>(
    data: ArrayViewD<A>,
    i: ArrayViewD<B>,
    attrs: &GatherAttrs,
) -> ArrayD<A> {
    let mut output_shape = data.shape().to_vec();
    let axis = if attrs.axis < 0 {
        (output_shape.len() as i64 + attrs.axis) as usize
    } else {
        attrs.axis as usize
    };
    output_shape.splice(axis..axis + 1, i.shape().iter().copied());
    let mut output = ndarray::ArrayD::zeros(output_shape);
    let n_i = &data.shape()[..axis];
    let n_k = &data.shape()[axis + 1..];
    let n_j = i.shape();
    for ii in NDIndex::new(n_i) {
        for jj in NDIndex::new(n_j) {
            for kk in NDIndex::new(n_k) {
                let slice_index = ii
                    .iter()
                    .chain(jj.iter().chain(kk.iter()))
                    .copied()
                    .collect::<Vec<_>>();
                let jjindex = [i[jj.as_slice()].as_()];
                //              ^^^^^^^^^^^^^^^
                // FIXME: Model GPT-2 crashes with index [0] being out of bounds for array of shape [1, 8] here
                let a_slice_index = ii
                    .iter()
                    .chain(jjindex.iter().chain(kk.iter()))
                    .copied()
                    .collect::<Vec<_>>();
                output[slice_index.as_slice()] = data[a_slice_index.as_slice()];
            }
        }
    }
    output
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_gather.py
/// https://onnx.ai/onnx/operators/onnx__Gather.html
pub fn gather(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    if inputs.len() != 2 {
        return Err(format!("Gather expects 2 inputs, got: {}", inputs.len()).into());
    }
    let data = inputs[0];
    let indices = inputs[1];
    let attrs = GatherAttrs::new(node);
    let rank = data.ndim() as i64;
    if attrs.axis < -rank || attrs.axis > rank {
        return Err(format!(
            "Gather axis must be in [-rank, rank-1], got: {}",
            attrs.axis
        )
        .into());
    }
    match indices {
        // FIXME:  All index values are expected to be within bounds [-s, s-1] along axis of size s
        ArrayType::I32(i) => {
            if i.is_empty() {
                if let ArrayType::F32(_) = data {
                    Ok(make_tensor_from_raw(&[1], &[], DataType::FLOAT.value())?.into())
                } else {
                    todo!("Gather for non-float data {}", data)
                }
            } else {
                match data {
                    ArrayType::F32(f32_data) => {
                        Ok(
                            ArrayType::F32(_gather_generic(f32_data.view(), i.view(), &attrs))
                                .into(),
                        )
                    }
                    ArrayType::I64(i64_data) => {
                        Ok(
                            ArrayType::I64(_gather_generic(i64_data.view(), i.view(), &attrs))
                                .into(),
                        )
                    }
                    data => todo!("Gather for non-float data {}", data),
                }
            }
        }
        ArrayType::I64(i) => {
            if i.is_empty() {
                if let ArrayType::F32(_) = data {
                    Ok(make_tensor_from_raw(&[1], &[], DataType::FLOAT.value())?.into())
                } else {
                    todo!("Gather for non-float data {}", data)
                }
            } else {
                match data {
                    ArrayType::F32(f32_data) => {
                        Ok(
                            ArrayType::F32(_gather_generic(f32_data.view(), i.view(), &attrs))
                                .into(),
                        )
                    }
                    ArrayType::I64(i64_data) => {
                        Ok(
                            ArrayType::I64(_gather_generic(i64_data.view(), i.view(), &attrs))
                                .into(),
                        )
                    }
                    data => todo!("Gather for non-float data {}", data),
                }
            }
        }
        _ => Err(format!("Gather expects indices to be I32 or I64, got: {}", indices).into()),
    }
}
