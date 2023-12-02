use anyhow::anyhow;
use ndarray::ArrayD;
use num::traits::AsPrimitive;
use protobuf::Enum;

use crate::common::{ArrayType, BoxResult, OperationResult};
use crate::onnx::{self, NodeProto};

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
                .map_or(DT::UNDEFINED, |a| {
                    a.i.map_or(DT::UNDEFINED, |v| {
                        DT::from_i32(v as i32).unwrap_or(DT::UNDEFINED)
                    })
                }),
        }
    }
}

type CastFn<'a, Out, In> = fn(In) -> Out;

fn cast_generic<
    Out: Clone + num::Zero + AsPrimitive<In>,
    In: Clone + num::Zero + AsPrimitive<Out>,
>(
    input: &ArrayD<In>,
    tfunc: Option<CastFn<Out, In>>,
) -> BoxResult<ArrayD<Out>> {
    if let Some(cfn) = tfunc {
        Ok(input.mapv(cfn))
    } else {
        Ok(input.mapv(|v| v.as_()))
    }
}

macro_rules! cast_impl {
    ($from:ty, $tgt:ident, $input:ident) => {
        match $input.data_type() {
            DataType::UNDEFINED => Err(anyhow!("Cast to undefined type")),
            DataType::FLOAT => {
                Ok(ArrayType::$tgt(cast_generic::<$from, f32>($input.try_into()?, None)?).into())
            }
            DataType::UINT8 => {
                Ok(ArrayType::$tgt(cast_generic::<$from, u8>($input.try_into()?, None)?).into())
            }
            DataType::INT8 => {
                Ok(ArrayType::$tgt(cast_generic::<$from, i8>($input.try_into()?, None)?).into())
            }
            DataType::UINT16 => {
                Ok(ArrayType::$tgt(cast_generic::<$from, u16>($input.try_into()?, None)?).into())
            }
            DataType::INT16 => {
                Ok(ArrayType::$tgt(cast_generic::<$from, i16>($input.try_into()?, None)?).into())
            }
            DataType::INT32 => {
                Ok(ArrayType::$tgt(cast_generic::<$from, i32>($input.try_into()?, None)?).into())
            }
            DataType::INT64 => {
                Ok(ArrayType::$tgt(cast_generic::<$from, i64>($input.try_into()?, None)?).into())
            }
            DataType::DOUBLE => {
                Ok(ArrayType::$tgt(cast_generic::<$from, f64>($input.try_into()?, None)?).into())
            }
            DataType::UINT32 => {
                Ok(ArrayType::$tgt(cast_generic::<$from, u32>($input.try_into()?, None)?).into())
            }
            DataType::UINT64 => {
                Ok(ArrayType::$tgt(cast_generic::<$from, u64>($input.try_into()?, None)?).into())
            }
            _ => Err(anyhow!(
                "Cast from {:?} to {} not supported",
                $input.data_type(),
                stringify!($ty)
            )),
        }
    };
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
    use onnx::tensor_proto::DataType;
    match attrs.to {
        DataType::UNDEFINED => Err(anyhow!("Cast to undefined type")),
        DataType::FLOAT => {
            cast_impl!(f32, F32, input)
        }
        DataType::UINT8 => {
            cast_impl!(u8, U8, input)
        }
        DataType::INT8 => {
            cast_impl!(i8, I8, input)
        }
        DataType::UINT16 => {
            cast_impl!(u16, U16, input)
        }
        DataType::INT16 => {
            cast_impl!(i16, I16, input)
        }
        DataType::INT32 => {
            cast_impl!(i32, I32, input)
        }
        DataType::INT64 => {
            cast_impl!(i64, I64, input)
        }
        DataType::DOUBLE => {
            cast_impl!(f64, F64, input)
        }
        DataType::UINT32 => {
            cast_impl!(u32, U32, input)
        }
        DataType::UINT64 => {
            cast_impl!(u64, U64, input)
        }
        _ => Err(anyhow!(
            "Cast from {:?} to {:?} not supported",
            input.data_type(),
            attrs.to
        )),
    }
}
