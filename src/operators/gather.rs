use protobuf::Enum;

use crate::{utils::{ArrayType, pick_opset_version, make_tensor}, onnx::{NodeProto, tensor_proto::DataType}};


const OPSET_VERSIONS: [i64; 3] = [1, 11, 13];

#[derive(Debug)]
struct GatherAttrs {
    axis: i64
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

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_gather.py
/// https://onnx.ai/onnx/operators/onnx__Gather.html

pub fn gather(
    inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    if inputs.len() != 2 {
        return Err(format!("Gather expects 2 inputs, got: {}", inputs.len()).into());
    }
    let data = inputs[0];
    let indices = inputs[1];
    let attrs = GatherAttrs::new(node);
    let rank = data.ndim() as i64;
    if attrs.axis < -rank || attrs.axis > rank {
        return Err(format!("Gather axis must be in [-rank, rank-1], got: {}", attrs.axis).into());
    }
    match indices {
        // FIXME:  All index values are expected to be within bounds [-s, s-1] along axis of size s
        ArrayType::I32(i) => {
            if i.len() == 0 {
                if let ArrayType::F32(_) = data {
                    make_tensor(&[1], &[], DataType::FLOAT.value())
                } else {
                    todo!("Gather for non-float data")
                }
            } else {
               match data {
                    ArrayType::F32(f32_data) => {
                        let mut result_data = Vec::with_capacity(i.len());
                        for &index in i.iter() {
                            result_data.push(f32_data[index as usize]);
                        }
                        // May be wrong
                        Ok(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[i.len()]), result_data).map(ArrayType::F32).unwrap())
                     }
                     _ => todo!("Gather for non-float data")
                
               }
            }
        }
        ArrayType::I64(i) => {
            if i.len() == 0 {
                if let ArrayType::F32(_) = data {
                    make_tensor(&[1], &[], DataType::FLOAT.value())
                } else {
                    todo!("Gather for non-float data")
                }
            } else {
                match data {
                    ArrayType::I64(i64_data) => {
                        let mut result_data = Vec::with_capacity(i.len());
                        for &index in i.iter() {
                            result_data.push(i64_data[index as usize]);
                        }
                        // May be wrong
                        Ok(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[i.len()]), result_data).map(ArrayType::I64).unwrap())
                    }
                    _ => todo!("Gather for non-int data")
              }
            }
        }
        _ => {
            return Err(format!("Gather expects indices to be I32 or I64, got: {}", indices).into());
        }
    }
}