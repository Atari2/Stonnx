use anyhow::anyhow;
use ndarray::{ArrayD, ArrayViewD};
use num::{traits::AsPrimitive, Zero};
use protobuf::Enum;

use crate::{
    common::{BoxResult, NDIndex, OperatorResult, TensorType},
    onnx::{tensor_proto::DataType, NodeProto},
    utils::make_tensor_from_raw,
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

/// Given data tensor of rank r >= 1, and indices tensor of rank q, gather entries of the axis dimension of data (by default outer-most one as axis=0) indexed by indices, and concatenates them in an output tensor of rank q + (r - 1).
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_gather.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__Gather.html)
pub fn gather(
    inputs: &[&TensorType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    if inputs.len() != 2 {
        return Err(anyhow!("Gather expects 2 inputs, got: {}", inputs.len()));
    }
    let data = inputs[0];
    let indices = inputs[1];
    let attrs = GatherAttrs::new(node);
    let rank = data.ndim() as i64;
    if attrs.axis < -rank || attrs.axis > rank {
        return Err(anyhow!(
            "Gather axis must be in [-rank, rank-1], got: {}",
            attrs.axis
        ));
    }
    match indices {
        TensorType::I32(i) => {
            if i.is_empty() {
                if let TensorType::F32(_) = data {
                    Ok(make_tensor_from_raw(&[1], &[], DataType::FLOAT.value())?.into())
                } else {
                    todo!("Gather for non-float data {}", data)
                }
            } else {
                match data {
                    TensorType::F32(f32_data) => {
                        Ok(
                            TensorType::F32(_gather_generic(f32_data.view(), i.view(), &attrs))
                                .into(),
                        )
                    }
                    TensorType::I64(i64_data) => {
                        Ok(
                            TensorType::I64(_gather_generic(i64_data.view(), i.view(), &attrs))
                                .into(),
                        )
                    }
                    data => todo!("Gather for non-float data {}", data),
                }
            }
        }
        TensorType::I64(i) => {
            if i.is_empty() {
                if let TensorType::F32(_) = data {
                    Ok(make_tensor_from_raw(&[1], &[], DataType::FLOAT.value())?.into())
                } else {
                    todo!("Gather for non-float data {}", data)
                }
            } else {
                match data {
                    TensorType::F32(f32_data) => {
                        Ok(
                            TensorType::F32(_gather_generic(f32_data.view(), i.view(), &attrs))
                                .into(),
                        )
                    }
                    TensorType::I64(i64_data) => {
                        Ok(
                            TensorType::I64(_gather_generic(i64_data.view(), i.view(), &attrs))
                                .into(),
                        )
                    }
                    data => todo!("Gather for non-float data {}", data),
                }
            }
        }
        _ => Err(anyhow!(
            "Gather expects indices to be I32 or I64, got: {}",
            indices
        )),
    }
}
