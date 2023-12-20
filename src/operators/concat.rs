use ndarray::{ArrayD, Axis, CowArray, IxDyn};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    common::{BoxResult, OperatorResult, TensorType, ValueType},
    onnx::NodeProto,
};
use anyhow::anyhow;

const _OPSET_VERSIONS: [i64; 4] = [1, 4, 11, 13];

#[derive(Debug)]
struct ConcatAttrs {
    axis: i64,
}

impl ConcatAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            axis: node
                .attribute
                .iter()
                .find(|a| a.name() == "axis")
                .map_or(1, |a| a.i.unwrap_or(1)),
        }
    }
}

fn _preprocess<A: Clone>(a: &ArrayD<A>, axis: i64) -> BoxResult<CowArray<'_, A, IxDyn>> {
    if a.shape().is_empty() {
        return Err(anyhow!("Input must be at least 1D"));
    }
    if axis >= a.shape().len() as i64 {
        let axis = axis as usize;
        let new_shape = a
            .shape()
            .iter()
            .chain(std::iter::repeat(&1).take(axis + 1 - a.shape().len()))
            .copied()
            .collect::<Vec<_>>();
        Ok(a.to_shape(new_shape)?.into_dyn()) // can fail is array not contiguous
    } else {
        Ok(a.into())
    }
}

fn _concat_i64(inputs: &[&TensorType], attrs: ConcatAttrs) -> BoxResult<TensorType> {
    let mut cow_array = vec![];
    inputs
        .par_iter()
        .map(|input| match input {
            TensorType::I64(x) => _preprocess(x, attrs.axis).expect("Preprocess for concat failed"),
            _ => panic!("Inputs not all of F32"),
        })
        .collect_into_vec(&mut cow_array);
    let inputs_i64 = cow_array
        .iter()
        .map(|array| array.view())
        .collect::<Vec<_>>();
    if inputs_i64.is_empty() {
        return Err(anyhow!("No inputs"));
    }
    let shape_i64 = inputs_i64[0].shape();
    let axis = if attrs.axis < 0 {
        shape_i64.len() as i64 + attrs.axis
    } else {
        attrs.axis
    } as usize;
    Ok(TensorType::I64(ndarray::concatenate(
        Axis(axis),
        &inputs_i64,
    )?))
}

fn _concat_f32(inputs: &[&TensorType], attrs: ConcatAttrs) -> BoxResult<TensorType> {
    let mut cow_array = vec![];
    inputs
        .par_iter()
        .map(|input| match input {
            TensorType::F32(x) => _preprocess(x, attrs.axis).expect("Preprocess for concat failed"),
            _ => panic!("Inputs not all of F32"),
        })
        .collect_into_vec(&mut cow_array);
    let inputs_f32 = cow_array
        .iter()
        .map(|array| array.view())
        .collect::<Vec<_>>();
    if inputs_f32.is_empty() {
        return Err(anyhow!("No inputs"));
    }
    let shape_f32 = inputs_f32[0].shape();
    let axis = if attrs.axis < 0 {
        shape_f32.len() as i64 + attrs.axis
    } else {
        attrs.axis
    } as usize;
    Ok(TensorType::F32(ndarray::concatenate(
        Axis(axis),
        &inputs_f32,
    )?))
}

/// Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_concat.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__Concat.html)
pub fn concat(
    inputs: &[&TensorType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let attrs = ConcatAttrs::new(node);

    if inputs.is_empty() {
        return Err(anyhow!("No inputs"));
    }

    let type_ = inputs[0].value_type();

    match type_ {
        ValueType::I64 => Ok(_concat_i64(inputs, attrs)?.into()),
        ValueType::F32 => Ok(_concat_f32(inputs, attrs)?.into()),
        _ => Err(anyhow!("Only f32 and i64 are supported")),
    }
}
