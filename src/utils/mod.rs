use anyhow::anyhow;
use num::traits::AsPrimitive;
use std::io::Read;
use std::os::raw::c_uchar;
use std::path::PathBuf;
use std::{collections::HashMap, io, path::Path};

use ndarray::{ArrayD, IxDyn};
use protobuf::Enum;

use crate::common::*;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::{TensorProto, ValueInfoProto};
use crate::onnxparser::onnx;
use crate::FileInputs;
use half::{bf16, f16};

pub fn shape_safe_product<
    'a,
    B: 'a + std::iter::Product<&'a B> + std::default::Default + Copy + 'static,
    A: IntoIterator<Item = &'a B>,
>(
    shape: A,
) -> B
where
    usize: AsPrimitive<B>,
{
    let mut piter = shape.into_iter().peekable();
    if piter.peek().is_none() {
        1_usize.as_()
    } else {
        piter.product()
    }
}

pub fn log_array_to_file<A: ndarray_npy::WritableElement, D: ndarray::Dimension>(
    operation: &str,
    name: &str,
    a: &ndarray::ArrayBase<ndarray::ViewRepr<&A>, D>,
) -> BoxResult<()> {
    let verbose_flag = VERBOSE.get();
    if let Some(4..) = verbose_flag {
        static mut COUNTER: usize = 0;
        unsafe {
            ndarray_npy::write_npy(
                format!(
                    "{}_intermediate_outputs/{}_{}.npy",
                    operation, COUNTER, name
                ),
                a,
            )?;
            COUNTER += 1;
        }
    }
    Ok(())
}

#[macro_export]
macro_rules! named_array_to_file {
    ($op:ident, $name:ident) => {{
        let $name = $name.view();
        $crate::utils::log_array_to_file(stringify!($op), stringify!($name), &$name).unwrap();
    }};
    ($op:ident, $var:ident, $name:expr) => {{
        let $var = $var.view();
        $crate::utils::log_array_to_file(stringify!($op), &$name, &$var).unwrap();
    }};
}

#[macro_export]
macro_rules! create_intermediate_output_dir_for {
    ($name:ident) => {{
        let verbose_flag = VERBOSE.get();
        if let Some(4..) = verbose_flag {
            match std::fs::create_dir(concat!(stringify!($name), "_intermediate_outputs")) {
                Ok(_) => {}
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::AlreadyExists {
                        return Err(anyhow!("Error creating rust_conv_outputs directory: {}", e));
                    }
                }
            }
        }
    }};
}

#[derive(Debug)]
pub struct ValueInfo {
    pub name: String,
    pub type_: (ValueType, Vec<i64>),
    pub doc_string: String,
}

#[derive(Debug)]
pub struct OutputInfo {
    pub valueinfo: ValueInfo,
    pub data: Option<ArrayType>,
}

impl OutputInfo {
    fn new(valueinfo: ValueInfo) -> Self {
        Self {
            valueinfo,
            data: None,
        }
    }
}

impl ValueInfo {
    fn from_proto(proto: &ValueInfoProto) -> BoxResult<Self> {
        if let Some(onnx::type_proto::Value::TensorType(tensor)) = &proto.type_.value {
            let dt = onnx::tensor_proto::DataType::from_i32(tensor.elem_type.unwrap_or_default())
                .unwrap_or_default();
            Ok(Self {
                name: proto
                    .name
                    .as_ref()
                    .map_or_else(|| UNKNOWN.to_owned(), |v| v.clone()),
                type_: (
                    ValueType::new(dt)?,
                    tensor.shape.dim.iter().map(|v| v.dim_value()).collect(),
                ),
                doc_string: proto
                    .doc_string
                    .as_ref()
                    .map_or_else(|| UNKNOWN.to_owned(), |v| v.clone()),
            })
        } else {
            todo!("ValueInfoProto type not supported: {:?}", proto.type_)
        }
    }
}

// FIXME: data in tensor may be external. Need to handle that.

pub fn make_tensor_from_proto(proto: &TensorProto) -> BoxResult<ArrayType> {
    let shape = &proto.dims;
    if proto.data_location() != onnx::tensor_proto::DataLocation::DEFAULT {
        return Err(anyhow!("External data location not supported"));
    }
    make_tensor(shape, proto, proto.data_type())
}

fn get_raw_data(proto: &TensorProto) -> BoxResult<(&[u8], usize)> {
    if let Some(ref raw_data) = proto.raw_data {
        Ok((raw_data.as_slice(), 1))
    } else if !proto.int32_data.is_empty() {
        Ok((
            bytemuck::try_cast_slice(proto.int32_data.as_slice()).map_err(|e| anyhow!(e))?,
            4,
        ))
    } else if !proto.int64_data.is_empty() {
        Ok((
            bytemuck::try_cast_slice(proto.int64_data.as_slice()).map_err(|e| anyhow!(e))?,
            8,
        ))
    } else if !proto.float_data.is_empty() {
        Ok((
            bytemuck::try_cast_slice(proto.float_data.as_slice()).map_err(|e| anyhow!(e))?,
            4,
        ))
    } else if !proto.double_data.is_empty() {
        Ok((
            bytemuck::try_cast_slice(proto.double_data.as_slice()).map_err(|e| anyhow!(e))?,
            8,
        ))
    } else if !proto.uint64_data.is_empty() {
        Ok((
            bytemuck::try_cast_slice(proto.uint64_data.as_slice()).map_err(|e| anyhow!(e))?,
            8,
        ))
    } else {
        Ok((&[], 0))
    }
}

pub fn make_tensor(shape: &[i64], proto: &TensorProto, data_type: i32) -> BoxResult<ArrayType> {
    let enum_dt = DataType::from_i32(data_type).unwrap_or_default();
    let shape = shape.iter().map(|v| *v as usize).collect::<Vec<usize>>();
    let (bytedata, origin_elem_size) = get_raw_data(proto)?;
    match enum_dt {
        DataType::UNDEFINED => Err(anyhow!("Undefined data type")),
        DataType::INT8 => match bytemuck::try_cast_slice::<u8, i8>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len() / origin_elem_size, shape_safe_product(&shape));
                let a = if origin_elem_size == std::mem::size_of::<i8>() {
                    ArrayD::<i8>::from_shape_vec(IxDyn(&shape), data.to_vec())?
                } else {
                    ArrayD::<i8>::from_shape_vec(
                        IxDyn(&shape),
                        data.iter().step_by(origin_elem_size).copied().collect(),
                    )?
                };
                Ok(ArrayType::I8(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::INT16 => match bytemuck::try_cast_slice::<u8, i16>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len() / origin_elem_size, shape_safe_product(&shape));
                let a = ArrayD::<i16>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I16(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::INT32 => {
            let data = if let Some(data) = &proto.raw_data {
                if data.is_empty() {
                    &[]
                } else {
                    match bytemuck::try_cast_slice::<u8, i32>(data) {
                        Ok(data) => data,
                        Err(e) => return Err(anyhow!(e)),
                    }
                }
            } else {
                proto.int32_data.as_slice()
            };
            let dlen = data.len();
            let slen = if !shape.is_empty() {
                shape_safe_product(&shape)
            } else {
                0
            };
            // if dlen != slen, check if data is 1 long and shape is [], then it is a scalar and it's fine
            // panic otherwise
            if dlen != slen && (slen == 0 && dlen != 1) {
                panic!("Data length {} does not match shape length {}", dlen, slen)
            }
            let a = if data.is_empty() {
                ArrayD::<i32>::zeros(IxDyn(&shape))
            } else {
                ArrayD::<i32>::from_shape_vec(IxDyn(&shape), data.to_vec())?
            };
            Ok(ArrayType::I32(a))
        }
        DataType::INT64 => {
            let data = if let Some(data) = &proto.raw_data {
                if data.is_empty() {
                    &[]
                } else {
                    match bytemuck::try_cast_slice::<u8, i64>(data) {
                        Ok(data) => data,
                        Err(e) => return Err(anyhow!(e)),
                    }
                }
            } else {
                proto.int64_data.as_slice()
            };
            let dlen = data.len();
            let slen = if !shape.is_empty() {
                shape_safe_product(&shape)
            } else {
                0
            };
            // if dlen != slen, check if data is 1 long and shape is [], then it is a scalar and it's fine
            // panic otherwise
            if dlen != slen && (slen == 0 && dlen != 1) {
                panic!("Data length {} does not match shape length {}", dlen, slen)
            }
            let a = if data.is_empty() {
                ArrayD::<i64>::zeros(IxDyn(&shape))
            } else {
                ArrayD::<i64>::from_shape_vec(IxDyn(&shape), data.to_vec())?
            };
            Ok(ArrayType::I64(a))
        }
        DataType::UINT8 => match bytemuck::try_cast_slice::<u8, u8>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len() / origin_elem_size, shape_safe_product(&shape));
                let a = if origin_elem_size == std::mem::size_of::<u8>() {
                    ArrayD::<u8>::from_shape_vec(IxDyn(&shape), data.to_vec())?
                } else {
                    ArrayD::<u8>::from_shape_vec(
                        IxDyn(&shape),
                        data.iter().step_by(origin_elem_size).copied().collect(),
                    )?
                };
                Ok(ArrayType::U8(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::UINT16 => match bytemuck::try_cast_slice::<u8, u16>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len() / origin_elem_size, shape_safe_product(&shape));
                let a = ArrayD::<u16>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U16(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::UINT32 => match bytemuck::try_cast_slice::<u8, u32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len() / origin_elem_size, shape_safe_product(&shape));
                let a = ArrayD::<u32>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U32(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::UINT64 => {
            let data = if let Some(data) = &proto.raw_data {
                if data.is_empty() {
                    &[]
                } else {
                    match bytemuck::try_cast_slice::<u8, u64>(data) {
                        Ok(data) => data,
                        Err(e) => return Err(anyhow!(e)),
                    }
                }
            } else {
                proto.uint64_data.as_slice()
            };
            let dlen = data.len();
            let slen = if !shape.is_empty() {
                shape_safe_product(&shape)
            } else {
                0
            };
            // if dlen != slen, check if data is 1 long and shape is [], then it is a scalar and it's fine
            // panic otherwise
            if dlen != slen && (slen == 0 && dlen != 1) {
                panic!("Data length {} does not match shape length {}", dlen, slen)
            }
            let a = if data.is_empty() {
                ArrayD::<u64>::zeros(IxDyn(&shape))
            } else {
                ArrayD::<u64>::from_shape_vec(IxDyn(&shape), data.to_vec())?
            };
            Ok(ArrayType::U64(a))
        }
        DataType::FLOAT16 => match bytemuck::try_cast_slice::<u8, u16>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len() / origin_elem_size, shape_safe_product(&shape));
                let a = ArrayD::<f16>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter().map(|x| f16::from_bits(*x)).collect(),
                )?;
                Ok(ArrayType::F16(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::BFLOAT16 => match bytemuck::try_cast_slice::<u8, f32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len() / origin_elem_size, shape_safe_product(&shape));
                let a = ArrayD::<bf16>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter().map(|x| bf16::from_f32(*x)).collect(),
                )?;
                Ok(ArrayType::BF16(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::DOUBLE => {
            let data = if let Some(data) = &proto.raw_data {
                if data.is_empty() {
                    &[]
                } else {
                    match bytemuck::try_cast_slice::<u8, f64>(data) {
                        Ok(data) => data,
                        Err(e) => return Err(anyhow!(e)),
                    }
                }
            } else {
                proto.double_data.as_slice()
            };
            let dlen = data.len();
            let slen = if !shape.is_empty() {
                shape_safe_product(&shape)
            } else {
                0
            };
            // if dlen != slen, check if data is 1 long and shape is [], then it is a scalar and it's fine
            // panic otherwise
            if dlen != slen && (slen == 0 && dlen != 1) {
                panic!("Data length {} does not match shape length {}", dlen, slen)
            }
            let a = if data.is_empty() {
                ArrayD::<f64>::zeros(IxDyn(&shape))
            } else {
                ArrayD::<f64>::from_shape_vec(IxDyn(&shape), data.to_vec())?
            };
            Ok(ArrayType::F64(a))
        }
        DataType::STRING => {
            let bytedata = &proto.string_data;
            let a = ArrayD::<String>::from_shape_vec(
                IxDyn(&shape),
                bytedata
                    .iter()
                    .map(|v| String::from_utf8_lossy(v.as_ref()).to_string())
                    .collect(),
            )?;
            Ok(ArrayType::Str(a))
        }
        DataType::BOOL => match bytemuck::try_cast_slice::<u8, c_uchar>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len() / origin_elem_size, shape_safe_product(&shape));
                let a = ArrayD::<bool>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter().map(|x| *x != 0).collect(),
                )?;
                Ok(ArrayType::Bool(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::FLOAT8E4M3FN
        | DataType::FLOAT8E4M3FNUZ
        | DataType::FLOAT8E5M2FNUZ
        | DataType::FLOAT8E5M2 => {
            todo!("Data type {:?} not supported", enum_dt);
        }
        DataType::FLOAT => {
            let data = if let Some(data) = &proto.raw_data {
                if data.is_empty() {
                    &[]
                } else {
                    match bytemuck::try_cast_slice::<u8, f32>(data) {
                        Ok(data) => data,
                        Err(e) => return Err(anyhow!(e)),
                    }
                }
            } else {
                proto.float_data.as_slice()
            };
            let dlen = data.len();
            let slen = if !shape.is_empty() {
                shape_safe_product(&shape)
            } else {
                0
            };
            // if dlen != slen, check if data is 1 long and shape is [], then it is a scalar and it's fine
            // panic otherwise
            if dlen != slen && (slen == 0 && dlen != 1) {
                panic!("Data length {} does not match shape length {}", dlen, slen)
            }
            let a = if data.is_empty() {
                ArrayD::<f32>::zeros(IxDyn(&shape))
            } else {
                ArrayD::<f32>::from_shape_vec(IxDyn(&shape), data.to_vec())?
            };
            Ok(ArrayType::F32(a))
        }
        DataType::COMPLEX64 => match bytemuck::try_cast_slice::<u8, Complex64Repr>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len() / origin_elem_size, shape_safe_product(&shape));
                let a = ArrayD::<Complex64>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter()
                        .map(|v| Complex64::new(v._val[0], v._val[1]))
                        .collect(),
                )?;
                Ok(ArrayType::C64(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::COMPLEX128 => match bytemuck::try_cast_slice::<u8, Complex128Repr>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len() / origin_elem_size, shape_safe_product(&shape));
                let a = ArrayD::<Complex128>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter()
                        .map(|v| Complex128::new(v._val[0], v._val[1]))
                        .collect(),
                )?;
                Ok(ArrayType::C128(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
    }
}

pub fn make_tensor_from_raw(
    shape: &[i64],
    bytedata: &[u8],
    data_type: i32,
) -> BoxResult<ArrayType> {
    let enum_dt = DataType::from_i32(data_type).unwrap_or_default();
    let shape = shape.iter().map(|v| *v as usize).collect::<Vec<usize>>();
    match enum_dt {
        DataType::UNDEFINED => Err(anyhow!("Undefined data type")),
        DataType::INT8 => match bytemuck::try_cast_slice::<u8, i8>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<i8>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I8(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::INT16 => match bytemuck::try_cast_slice::<u8, i16>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<i16>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I16(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::INT32 => match bytemuck::try_cast_slice::<u8, i32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<i32>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I32(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::INT64 => match bytemuck::try_cast_slice::<u8, i64>(bytedata) {
            Ok(data) => {
                let dlen = data.len();
                let slen = if !shape.is_empty() {
                    shape_safe_product(&shape)
                } else {
                    0
                };
                // if dlen != slen, check if data is 1 long and shape is [], then it is a scalar and it's fine
                // panic otherwise
                if dlen != slen && (slen == 0 && dlen != 1) {
                    panic!("Data length {} does not match shape length {}", dlen, slen)
                }
                let a = if data.is_empty() {
                    ArrayD::<i64>::zeros(IxDyn(&shape))
                } else {
                    ArrayD::<i64>::from_shape_vec(IxDyn(&shape), data.to_vec())?
                };
                Ok(ArrayType::I64(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::UINT8 => match bytemuck::try_cast_slice::<u8, u8>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<u8>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U8(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::UINT16 => match bytemuck::try_cast_slice::<u8, u16>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<u16>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U16(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::UINT32 => match bytemuck::try_cast_slice::<u8, u32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<u32>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U32(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::UINT64 => match bytemuck::try_cast_slice::<u8, u64>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<u64>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U64(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::FLOAT16 => match bytemuck::try_cast_slice::<u8, u16>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<f16>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter().map(|x| f16::from_bits(*x)).collect(),
                )?;
                Ok(ArrayType::F16(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::BFLOAT16 => match bytemuck::try_cast_slice::<u8, f32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<bf16>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter().map(|x| bf16::from_f32(*x)).collect(),
                )?;
                Ok(ArrayType::BF16(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::DOUBLE => match bytemuck::try_cast_slice::<u8, f64>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<f64>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::F64(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::STRING => panic!("String data type not supported, use make_string_tensor()"),
        DataType::BOOL => match bytemuck::try_cast_slice::<u8, c_uchar>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<bool>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter().map(|x| *x != 0).collect(),
                )?;
                Ok(ArrayType::Bool(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::FLOAT
        | DataType::FLOAT8E4M3FN
        | DataType::FLOAT8E4M3FNUZ
        | DataType::FLOAT8E5M2FNUZ
        | DataType::FLOAT8E5M2 => match bytemuck::try_cast_slice::<u8, f32>(bytedata) {
            Ok(data) => {
                let dlen = data.len();
                let slen = if !shape.is_empty() {
                    shape_safe_product(&shape)
                } else {
                    0
                };
                // if dlen != slen, check if data is 1 long and shape is [], then it is a scalar and it's fine
                // panic otherwise
                if dlen != slen && (slen == 0 && dlen != 1) {
                    panic!("Data length {} does not match shape length {}", dlen, slen)
                }
                let a = if data.is_empty() {
                    ArrayD::<f32>::zeros(IxDyn(&shape))
                } else {
                    ArrayD::<f32>::from_shape_vec(IxDyn(&shape), data.to_vec())?
                };
                Ok(ArrayType::F32(a))
            }
            Err(e) => {
                println!("Copying data of tensor as f32 because {}", e);
                let mut copied_data = vec![];
                for float_slice in bytedata.chunks_exact(std::mem::size_of::<f32>()) {
                    copied_data.push(f32::from_le_bytes(float_slice.try_into()?));
                }
                let a = ArrayD::<f32>::from_shape_vec(IxDyn(&shape), copied_data)?;
                Ok(ArrayType::F32(a))
            }
        },
        DataType::COMPLEX64 => match bytemuck::try_cast_slice::<u8, Complex64Repr>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<Complex64>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter()
                        .map(|v| Complex64::new(v._val[0], v._val[1]))
                        .collect(),
                )?;
                Ok(ArrayType::C64(a))
            }
            Err(e) => Err(anyhow!(e)),
        },
        DataType::COMPLEX128 => match bytemuck::try_cast_slice::<u8, Complex128Repr>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<Complex128>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter()
                        .map(|v| Complex128::new(v._val[0], v._val[1]))
                        .collect(),
                )?;
                Ok(ArrayType::C128(a))
            }
            Err(e) => Err(anyhow!(e.to_string())),
        },
    }
}

pub fn make_initializers(graph: &onnx::GraphProto) -> HashMap<String, ArrayType> {
    let mut initializers: HashMap<String, ArrayType> = HashMap::new();
    for tensor in graph.initializer.iter() {
        let tensor_name = tensor.name.as_ref().map_or(UNKNOWN, |v| v.as_str());
        if !tensor.has_data_type() {
            println!("  Tensor: {} has no data type", tensor_name);
        } else {
            match make_tensor_from_proto(tensor) {
                Ok(a) => {
                    initializers.insert(tensor_name.to_string(), a);
                }
                Err(e) => {
                    panic!(
                        "  Tensor: {} has data type {:?} but error: {}",
                        tensor_name,
                        tensor.data_type(),
                        e
                    );
                }
            }
        }
    }
    initializers
}

fn make_input_tensors_from_files(
    graph: &onnx::GraphProto,
    files: &[PathBuf],
    initializers: &HashMap<String, ArrayType>,
) -> BoxResult<HashMap<String, ArrayType>> {
    let mut map = HashMap::new();
    let mut external_inputs_map = HashMap::new();
    for input in files.iter() {
        let input_tensor = read_tensor(input)?;
        external_inputs_map.insert(
            input_tensor
                .name
                .as_ref()
                .map_or_else(|| UNKNOWN.to_owned(), |v| v.clone()),
            input_tensor,
        );
    }
    for input in graph.input.iter() {
        let input_name = input.name.as_ref().map_or(UNKNOWN, |v| v.as_str());
        if let Some(input_from_file) = external_inputs_map.get(input_name) {
            let tensor = make_tensor_from_proto(input_from_file)?;
            println!(
                "  Input {} from file has shape {:?} and type {:?}",
                input_name,
                tensor.shape(),
                tensor.value_type()
            );
            map.insert(input_name.to_string(), tensor);
        } else if initializers.get(input_name).is_none() {
            return Err(anyhow!(
                "Input {} not found in inputs file or graph initializers",
                input_name
            ));
        }
    }
    Ok(map)
}

fn make_output_tensors_from_files(
    graph: &onnx::GraphProto,
    files: &[PathBuf],
) -> BoxResult<HashMap<String, ArrayType>> {
    let mut map = HashMap::new();
    let mut external_outputs_map = HashMap::new();
    for output in files.iter() {
        let ouput_tensor = read_tensor(output)?;
        external_outputs_map.insert(
            ouput_tensor
                .name
                .as_ref()
                .map_or_else(|| UNKNOWN.to_owned(), |v| v.clone()),
            ouput_tensor,
        );
    }
    for output in graph.output.iter() {
        let output_name = output.name.as_ref().map_or(UNKNOWN, |v| v.as_str());
        if let Some(output_from_file) = external_outputs_map.get(output_name) {
            map.insert(
                output_name.to_string(),
                make_tensor_from_proto(output_from_file)?,
            );
        } else {
            return Err(anyhow!("Output {} not found in inputs file", output_name));
        }
    }
    Ok(map)
}

pub fn make_external_inputs(
    graph: &onnx::GraphProto,
    fileinputs: &FileInputs,
    initializers: &HashMap<String, ArrayType>,
) -> BoxResult<HashMap<String, ArrayType>> {
    if fileinputs.inputs.is_empty() {
        return Ok(HashMap::new());
    }
    make_input_tensors_from_files(graph, &fileinputs.inputs, initializers)
}

pub fn make_external_outputs(
    graph: &onnx::GraphProto,
    fileinputs: &FileInputs,
) -> BoxResult<HashMap<String, ArrayType>> {
    if fileinputs.outputs.is_empty() {
        return Ok(HashMap::new());
    }
    make_output_tensors_from_files(graph, &fileinputs.outputs)
}

pub fn make_graph_outputs(graph: &onnx::GraphProto) -> BoxResult<HashMap<String, OutputInfo>> {
    let mut map = HashMap::new();
    for output in graph.output.iter() {
        let output_name = output.name.as_ref().map_or(UNKNOWN, |v| v.as_str());
        map.insert(
            output_name.to_string(),
            OutputInfo::new(ValueInfo::from_proto(output)?),
        );
    }
    Ok(map)
}

fn read_model_text(p: &Path) -> BoxResult<onnx::ModelProto> {
    let file = std::fs::File::open(p)?;
    let mut reader = io::BufReader::new(file);
    let mut buf = String::new();
    reader.read_to_string(&mut buf)?;
    let model = protobuf::text_format::parse_from_str(&buf)?;
    Ok(model)
}
fn read_model_binary(p: &Path) -> BoxResult<onnx::ModelProto> {
    let file = std::fs::File::open(p)?;
    let mut reader = io::BufReader::new(file);
    let model: onnx::ModelProto = protobuf::Message::parse_from_reader(&mut reader)?;
    Ok(model)
}

pub fn read_model(p: &Path) -> BoxResult<onnx::ModelProto> {
    println!("Reading model from {}", p.display());
    let merr = read_model_binary(p);
    match merr {
        Ok(m) => Ok(m),
        Err(e) => {
            eprintln!("Error reading binary model: {}", e);
            read_model_text(p)
        }
    }
}

pub fn read_tensor(p: &Path) -> BoxResult<onnx::TensorProto> {
    let file = std::fs::File::open(p)?;
    let mut reader = io::BufReader::new(file);
    let model: onnx::TensorProto = protobuf::Message::parse_from_reader(&mut reader)?;
    Ok(model)
}

pub fn pick_opset_version(target_ver: i64, opset_versions: &[i64]) -> i64 {
    let mut opset_version = 0;
    for v in opset_versions.iter() {
        if *v <= target_ver && *v > opset_version {
            opset_version = *v;
        }
    }
    opset_version
}
