use crate::common::{BoxResult, OperatorResult, TensorType};
use crate::onnx::NodeProto;
use anyhow::anyhow;
use ndarray::{ArrayD, SliceInfoElem};

const _OPSET_VERSIONS: [i64; 5] = [1, 2, 11, 13, 18];

#[derive(Debug)]
struct SplitAttrs<'a> {
    axis: i64,
    split: &'a [i64],
    num_outputs: Option<i64>,
}

impl<'a> SplitAttrs<'a> {
    fn new(node: &'a NodeProto) -> Self {
        Self {
            axis: node
                .attribute
                .iter()
                .find(|a| a.name() == "axis")
                .map_or(0, |a| a.i.unwrap_or(0)),
            split: node
                .attribute
                .iter()
                .find(|a| a.name() == "split")
                .map_or(&[], |a| a.ints.as_slice()),
            num_outputs: node
                .attribute
                .iter()
                .find(|a| a.name() == "num_outputs")
                .map(|a| a.i.unwrap_or(0)),
        }
    }
}

fn _split_part_generic<A: Copy>(
    mat: &ArrayD<A>,
    split: ArrayD<i64>,
    axis: usize,
) -> Vec<ArrayD<A>> {
    let mut sli = mat
        .shape()
        .iter()
        .map(|s| (0..*s).into())
        .collect::<Vec<SliceInfoElem>>();
    let mut pos = 0;
    let mut res = vec![];
    for spl in split {
        sli[axis] = (pos..(pos + spl as usize)).into();
        pos += spl as usize;
        res.push(mat.slice(sli.as_slice()).to_owned());
    }
    res
}

fn split_impl(
    mat: &TensorType,
    split: Option<&TensorType>,
    attrs: SplitAttrs,
    output_len: usize,
) -> BoxResult<Vec<TensorType>> {
    let n_outputs = attrs.num_outputs.unwrap_or(output_len as i64) as usize;
    let axis = if attrs.axis < 0 {
        mat.ndim() as i64 + attrs.axis
    } else {
        attrs.axis
    } as usize;
    let split = match split {
        Some(TensorType::I64(s)) => Some(s.clone()),
        None => Some(ArrayD::<i64>::from_shape_vec(
            vec![attrs.split.len()],
            attrs.split.to_vec(),
        )?),
        _ => return Err(anyhow!("Split must be an array of i64")),
    };
    let split = if let Some(split) = split {
        split
    } else if mat.shape()[axis] % n_outputs == 0 {
        let div = mat.shape()[axis] / n_outputs;
        ArrayD::<i64>::from_shape_vec(
            vec![n_outputs],
            std::iter::repeat(div as i64).take(n_outputs).collect(),
        )?
    } else {
        let div = mat.shape()[axis] / n_outputs + 1;
        let mut ret = ArrayD::<i64>::from_shape_vec(
            vec![n_outputs],
            std::iter::repeat(div as i64).take(n_outputs).collect(),
        )?;
        let last = ret.len() - 1;
        ret[last] += (mat.shape()[axis] % n_outputs) as i64;
        ret
    };

    match mat {
        TensorType::F32(mat) => Ok(_split_part_generic(mat, split, axis)
            .into_iter()
            .map(TensorType::F32)
            .collect()),
        TensorType::I64(mat) => Ok(_split_part_generic(mat, split, axis)
            .into_iter()
            .map(TensorType::I64)
            .collect()),
        _ => {
            todo!("Split for type {}", mat)
        }
    }
}

/// Split a tensor into a list of tensors, along the specified ‘axis’.
///
/// Either input ‘split’ or the attribute ‘num_outputs’ should be specified, but not both.
///
/// If the attribute ‘num_outputs’ is specified, then the tensor is split into equal sized parts.
/// If the tensor is not evenly splittable into num_outputs, the last chunk will be smaller.
/// If the input ‘split’ is specified, it indicates the sizes of each output in the split.
///
/// [Python reference](<https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_split.py>)
///
/// [ONNX Documentation](<https://onnx.ai/onnx/operators/onnx__Split.html>)
pub fn split(
    inputs: &[&TensorType],
    node: &NodeProto,
    _opset_version: i64,
    output_len: usize,
) -> BoxResult<OperatorResult> {
    let mat = inputs[0];
    let split = inputs.get(1);
    let attrs = SplitAttrs::new(node);
    Ok(split_impl(mat, split.copied(), attrs, output_len)?.into())
}
