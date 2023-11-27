use crate::{
    onnx::{tensor_proto::DataType, NodeProto},
    utils::{
        make_tensor_from_proto, make_tensor_from_raw, pick_opset_version, ArrayType, BoxResult,
        OperationResult,
    },
};
use anyhow::anyhow;
use ndarray::{Array0, Array1};
use protobuf::{Enum, MessageField};

const OPSET_VERSION: [i64; 6] = [1, 9, 11, 12, 13, 19];

#[derive(Debug)]
struct ConstantAttrs {
    sparse_value: Option<f32>,
    value: Option<ArrayType>,
    value_float: Option<f32>,
    value_floats: Option<Vec<f32>>,
    value_int: Option<i64>,
    value_ints: Option<Vec<i64>>,
    value_string: Option<String>,
    value_strings: Option<Vec<String>>,
}

impl ConstantAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            sparse_value: None, // to be implemented
            value: node
                .attribute
                .iter()
                .find(|a| a.name() == "value")
                .and_then(|a| match &a.t {
                    MessageField(Some(t)) => match make_tensor_from_proto(t) {
                        Ok(t) => Some(t),
                        Err(e) => {
                            panic!("Error while parsing constant attributes: {:?}", e)
                        }
                    },
                    MessageField(None) => None,
                }),
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
        }
    }
}

fn constant_1(attrs: ConstantAttrs) -> BoxResult<ArrayType> {
    let value = attrs.value.ok_or(anyhow!(
        "Constant_1 operator requires the 'value' attribute"
    ))?;
    Ok(value)
}

fn constant_9(attrs: ConstantAttrs) -> BoxResult<ArrayType> {
    constant_1(attrs)
}

fn constant_11(attrs: ConstantAttrs) -> BoxResult<ArrayType> {
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

fn constant_12(attrs: ConstantAttrs) -> BoxResult<ArrayType> {
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
            Ok(ArrayType::Str(Array0::from_elem([], v_s).into_dyn()))
        }
        (None, None, None, None, None, None, None, Some(v_ss)) => {
            Ok(ArrayType::Str(Array1::from_vec(v_ss).into_dyn()))
        }
        _ => todo!(),
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_constant.py
/// https://onnx.ai/onnx/operators/onnx__Constant.html
pub fn constant(
    _inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSION);
    if target_version == 1 {
        Ok(constant_1(ConstantAttrs::new(node))?.into())
    } else if target_version == 9 {
        Ok(constant_9(ConstantAttrs::new(node))?.into())
    } else if target_version == 11 {
        Ok(constant_11(ConstantAttrs::new(node))?.into())
    } else {
        Ok(constant_12(ConstantAttrs::new(node))?.into())
    }
}
