use crate::{
    common::{BoxResult, OperatorResult, SparseTensorType, TensorType},
    onnx::{tensor_proto::DataType, NodeProto},
    utils::{make_tensor_from_proto, make_tensor_from_raw, pick_opset_version},
};
use anyhow::anyhow;
use ndarray::{Array0, Array1};
use protobuf::{Enum, MessageField};

const OPSET_VERSION: [i64; 6] = [1, 9, 11, 12, 13, 19];

#[derive(Debug)]
struct ConstantAttrs {
    sparse_value: Option<SparseTensorType>,
    value: Option<TensorType>,
    value_float: Option<f32>,
    value_floats: Option<Vec<f32>>,
    value_int: Option<i64>,
    value_ints: Option<Vec<i64>>,
    value_string: Option<String>,
    value_strings: Option<Vec<String>>,
}

impl ConstantAttrs {
    fn new(node: &NodeProto) -> BoxResult<Self> {
        Ok(Self {
            sparse_value: node
                .attribute
                .iter()
                .find(|a| a.name() == "sparse_value")
                .and_then(|a| match &a.t {
                    MessageField(Some(_)) => unimplemented!("OP Constant sparse_value"),
                    MessageField(None) => None,
                }),
            value: node.attribute.iter().find(|a| a.name() == "value").map_or(
                Ok::<Option<TensorType>, anyhow::Error>(None),
                |a| match &a.t {
                    MessageField(Some(t)) => Ok(Some(make_tensor_from_proto(t)?)),
                    MessageField(None) => Ok(None),
                },
            )?,
            value_float: node
                .attribute
                .iter()
                .find(|a| a.name() == "value_float")
                .and_then(|a| a.f),
            value_floats: node
                .attribute
                .iter()
                .find(|a| a.name() == "value_floats")
                .map(|a| a.f.iter().copied().collect()),
            value_int: node
                .attribute
                .iter()
                .find(|a| a.name() == "value_int")
                .and_then(|a| a.i),
            value_ints: node
                .attribute
                .iter()
                .find(|a| a.name() == "value_ints")
                .map(|a| a.ints.to_vec()),
            value_string: node
                .attribute
                .iter()
                .find(|a| a.name() == "value_string")
                .and_then(|a| {
                    a.s.clone()
                        .as_ref()
                        .map(|s| String::from_utf8_lossy(s).to_string())
                }),
            value_strings: node
                .attribute
                .iter()
                .find(|a| a.name() == "value_strings")
                .map(|a| {
                    a.strings
                        .iter()
                        .map(|s| String::from_utf8_lossy(s).to_string())
                        .collect()
                }),
        })
    }
}

fn constant_1(attrs: ConstantAttrs) -> BoxResult<TensorType> {
    let value = attrs.value.ok_or(anyhow!(
        "Constant_1 operator requires the 'value' attribute"
    ))?;
    Ok(value)
}

fn constant_9(attrs: ConstantAttrs) -> BoxResult<TensorType> {
    constant_1(attrs)
}

fn constant_11(attrs: ConstantAttrs) -> BoxResult<TensorType> {
    let value = attrs.value;
    let sparse_value = attrs.sparse_value;

    match (sparse_value, value) {
        (Some(_), None) => {
            todo!("Constant_11 sparse_value")
        }
        (None, Some(value)) => Ok(value),
        _ => Err(anyhow!(
            "Constant_11 operator requires either 'sparse_value' or 'value' attribute, not both"
        )),
    }
}

fn constant_12(attrs: ConstantAttrs) -> BoxResult<TensorType> {
    let v = attrs.value;
    let s_v = attrs.sparse_value;
    let v_f = attrs.value_float;
    let v_fs = attrs.value_floats;
    let v_i = attrs.value_int;
    let v_is = attrs.value_ints;
    let v_s = attrs.value_string;
    let v_ss = attrs.value_strings;

    // only one of these 3: value, sparse_value or value_*
    match (v, s_v, v_f, v_fs, v_i, v_is, v_s, v_ss) {
        (Some(v), None, None, None, None, None, None, None) => Ok(v),
        (None, Some(_), None, None, None, None, None, None) => todo!(),
        (None, None, Some(v_f), None, None, None, None, None) => {
            let float_value = v_f.to_le_bytes();
            Ok(make_tensor_from_raw(
                &[],
                &float_value,
                DataType::FLOAT.value(),
            )?)
        }
        (None, None, None, Some(v_fs), None, None, None, None) => {
            let float_value: Vec<u8> = v_fs.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
            Ok(make_tensor_from_raw(
                &[v_fs.len() as i64],
                &float_value,
                DataType::FLOAT.value(),
            )?)
        }
        (None, None, None, None, Some(v_i), None, None, None) => {
            let int_value = v_i.to_le_bytes();
            Ok(make_tensor_from_raw(
                &[],
                &int_value,
                DataType::INT64.value(),
            )?)
        }
        (None, None, None, None, None, Some(v_is), None, None) => {
            let int_value: Vec<u8> = v_is.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
            Ok(make_tensor_from_raw(
                &[v_is.len() as i64],
                &int_value,
                DataType::INT64.value(),
            )?)
        }
        (None, None, None, None, None, None, Some(v_s), None) => {
            Ok(TensorType::Str(Array0::from_elem([], v_s).into_dyn()))
        }
        (None, None, None, None, None, None, None, Some(v_ss)) => {
            Ok(TensorType::Str(Array1::from_vec(v_ss).into_dyn()))
        }
        _ => todo!(),
    }
}

/// This operator produces a constant tensor.
///
/// Exactly one of the provided attributes, either value, sparse_value, or value_* must be specified.
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_constant.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__Constant.html)
pub fn constant(
    _inputs: &[&TensorType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSION);
    if target_version == 1 {
        Ok(constant_1(ConstantAttrs::new(node)?)?.into())
    } else if target_version == 9 {
        Ok(constant_9(ConstantAttrs::new(node)?)?.into())
    } else if target_version == 11 {
        Ok(constant_11(ConstantAttrs::new(node)?)?.into())
    } else {
        Ok(constant_12(ConstantAttrs::new(node)?)?.into())
    }
}
