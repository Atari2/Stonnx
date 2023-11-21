use protobuf::Enum;

use crate::onnx::{NodeProto, self};
use crate::utils::{ArrayType, BoxResult, OperationResult};

const _OPSET_VERSIONS: [i64; 5] = [1, 6, 9, 13, 19];

#[derive(Debug)]
struct CastAttrs {
    // saturate only applies to float 8 types (i.e. float8e4m3fn which we are not supporting currently)
    _saturate: i64,
    to: onnx::tensor_proto::DataType,
}

impl CastAttrs {
    pub fn new(node: &NodeProto) -> Self {
        type DT = onnx::tensor_proto::DataType;
        Self {
            _saturate: node
                .attribute
                .iter()
                .find(|a| a.name() == "saturate")
                .map_or(1, |a| a.i.unwrap_or(1)),
            to: node
                .attribute
                .iter()
                .find(|a| a.name() == "to")
                .map_or(DT::UNDEFINED, |a| a.i.map_or(DT::UNDEFINED, |v| DT::from_i32(v as i32).unwrap_or(DT::UNDEFINED))),
        }
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_cast.py
/// https://onnx.ai/onnx/operators/onnx__Cast.html
pub fn cast(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let input = inputs[0];
    let attrs = CastAttrs::new(_node);
    if attrs.to == input.data_type() {
        // No-op
        return Ok(input.to_owned().into());
    }
    todo!("Cast")
}
