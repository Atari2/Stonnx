use std::io::Read;
use std::os::raw::c_uchar;
use std::{collections::HashMap, error::Error, io, path::Path};

use ndarray::{ArrayD, IxDyn};
use protobuf::Enum;

use crate::onnxparser::onnx;
use crate::onnxparser::onnx::tensor_proto;
use crate::onnxparser::onnx::tensor_shape_proto::Dimension;
use crate::{FileInput, FileInputs};
use half::{bf16, f16};
use num::complex::Complex;

type Complex64 = Complex<f32>;
type Complex128 = Complex<f64>;

static UNKNOWN: &str = "<unknown>";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
    C64,
    C128,
    String,
    Bool,
}

#[derive(Debug, Clone)]
pub enum ArrayType {
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    U16(ArrayD<u16>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
    F16(ArrayD<f16>),
    BF16(ArrayD<bf16>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    C64(ArrayD<Complex64>),
    C128(ArrayD<Complex128>),
    Str(ArrayD<String>),
    Bool(ArrayD<bool>),
}

/*
typedef struct
{
    double _Val[2];
} npy_cdouble;

#define NPY_COMPLEX128 NPY_CDOUBLE
        typedef double npy_float64;
        typedef npy_cdouble npy_complex128;

typedef struct
{
    float _Val[2];
} npy_cfloat;

#define NPY_COMPLEX64 NPY_CFLOAT
        typedef float npy_float32;
        typedef npy_cfloat npy_complex64;
*/

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
struct Complex64Repr {
    _val: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
struct Complex128Repr {
    _val: [f64; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
struct HalfFloat {
    _val: u16,
}

impl std::fmt::Display for ArrayType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrayType::I8(_) => write!(f, "I8"),
            ArrayType::I16(_) => write!(f, "I16"),
            ArrayType::I32(_) => write!(f, "I32"),
            ArrayType::U8(_) => write!(f, "U8"),
            ArrayType::U16(_) => write!(f, "U16"),
            ArrayType::U32(_) => write!(f, "U32"),
            ArrayType::I64(_) => write!(f, "I64"),
            ArrayType::U64(_) => write!(f, "U64"),
            ArrayType::F16(_) => write!(f, "F16"),
            ArrayType::BF16(_) => write!(f, "BF16"),
            ArrayType::F32(_) => write!(f, "F32"),
            ArrayType::F64(_) => write!(f, "F64"),
            ArrayType::C64(_) => write!(f, "C64"),
            ArrayType::C128(_) => write!(f, "C128"),
            ArrayType::Str(_) => write!(f, "STR"),
            ArrayType::Bool(_) => write!(f, "BOOL"),
        }
    }
}

impl ArrayType {
    pub fn shape(&self) -> &[usize] {
        match self {
            ArrayType::I8(a) => a.shape(),
            ArrayType::I16(a) => a.shape(),
            ArrayType::I32(a) => a.shape(),
            ArrayType::I64(a) => a.shape(),
            ArrayType::U8(a) => a.shape(),
            ArrayType::U16(a) => a.shape(),
            ArrayType::U32(a) => a.shape(),
            ArrayType::U64(a) => a.shape(),
            ArrayType::F16(a) => a.shape(),
            ArrayType::BF16(a) => a.shape(),
            ArrayType::F32(a) => a.shape(),
            ArrayType::F64(a) => a.shape(),
            ArrayType::C64(a) => a.shape(),
            ArrayType::C128(a) => a.shape(),
            ArrayType::Str(a) => a.shape(),
            ArrayType::Bool(a) => a.shape(),
        }
    }
    pub fn value_type(&self) -> ValueType {
        match self {
            ArrayType::I64(_) => ValueType::I64,
            ArrayType::F32(_) => ValueType::F32,
            ArrayType::I8(_) => ValueType::I8,
            ArrayType::I16(_) => ValueType::I16,
            ArrayType::I32(_) => ValueType::I32,
            ArrayType::U8(_) => ValueType::U8,
            ArrayType::U16(_) => ValueType::U16,
            ArrayType::U32(_) => ValueType::U32,
            ArrayType::U64(_) => ValueType::U64,
            ArrayType::F16(_) => ValueType::F16,
            ArrayType::BF16(_) => ValueType::BF16,
            ArrayType::F64(_) => ValueType::F64,
            ArrayType::C64(_) => ValueType::C64,
            ArrayType::C128(_) => ValueType::C128,
            ArrayType::Str(_) => ValueType::String,
            ArrayType::Bool(_) => ValueType::Bool,
        }
    }
}

pub fn make_tensor(
    shape: &[i64],
    bytedata: &[u8],
    data_type: i32,
) -> Result<ArrayType, Box<dyn Error>> {
    use tensor_proto::DataType;
    let enum_dt = DataType::from_i32(data_type).unwrap_or_default();
    let shape = shape.iter().map(|v| *v as usize).collect::<Vec<usize>>();
    match enum_dt {
        DataType::UNDEFINED => Err("Undefined data type".into()),
        DataType::INT8 => match bytemuck::try_cast_slice::<u8, i8>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<i8>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I8(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::INT16 => match bytemuck::try_cast_slice::<u8, i16>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<i16>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I16(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::INT32 => match bytemuck::try_cast_slice::<u8, i32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<i32>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I32(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::INT64 => match bytemuck::try_cast_slice::<u8, i64>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<i64>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I64(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::UINT8 => match bytemuck::try_cast_slice::<u8, u8>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<u8>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U8(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::UINT16 => match bytemuck::try_cast_slice::<u8, u16>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<u16>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U16(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::UINT32 => match bytemuck::try_cast_slice::<u8, u32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<u32>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U32(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::UINT64 => match bytemuck::try_cast_slice::<u8, u64>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<u64>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U64(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::FLOAT16 => match bytemuck::try_cast_slice::<u8, u16>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<f16>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter().map(|x| f16::from_bits(*x)).collect(),
                )?;
                Ok(ArrayType::F16(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::BFLOAT16 => match bytemuck::try_cast_slice::<u8, f32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<bf16>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter().map(|x| bf16::from_f32(*x)).collect(),
                )?;
                Ok(ArrayType::BF16(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::DOUBLE => match bytemuck::try_cast_slice::<u8, f64>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<f64>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::F64(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::STRING => todo!("String data type not implemented"),
        DataType::BOOL => match bytemuck::try_cast_slice::<u8, c_uchar>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<bool>::from_shape_vec(IxDyn(&shape), data.iter().map(|x| *x != 0).collect())?;
                Ok(ArrayType::Bool(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::FLOAT
        | DataType::FLOAT8E4M3FN
        | DataType::FLOAT8E4M3FNUZ
        | DataType::FLOAT8E5M2FNUZ
        | DataType::FLOAT8E5M2 => match bytemuck::try_cast_slice::<u8, f32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<f32>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::F32(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::COMPLEX64 => match bytemuck::try_cast_slice::<u8, Complex64Repr>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<Complex64>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter()
                        .map(|v| Complex64::new(v._val[0], v._val[1]))
                        .collect(),
                )?;
                Ok(ArrayType::C64(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::COMPLEX128 => match bytemuck::try_cast_slice::<u8, Complex128Repr>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayD::<Complex128>::from_shape_vec(
                    IxDyn(&shape),
                    data.iter()
                        .map(|v| Complex128::new(v._val[0], v._val[1]))
                        .collect(),
                )?;
                Ok(ArrayType::C128(a))
            }
            Err(e) => Err(e.to_string().into()),
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
            match make_tensor(&tensor.dims, tensor.raw_data(), tensor.data_type()) {
                Ok(a) => {
                    initializers.insert(tensor_name.to_string(), a);
                }
                Err(e) => {
                    println!(
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

pub fn get_dim(dim: &Dimension, file_input: Option<&FileInput>) -> i64 {
    if dim.has_dim_value() {
        dim.dim_value()
    } else if let Some(onnx::tensor_shape_proto::dimension::Value::DimParam(p)) = &dim.value {
        if let Some(attr) = file_input.and_then(|i| i.attributes.get(p)) {
            match attr {
                serde_json::Value::Number(n) => n.as_i64().unwrap_or_default(),
                _ => panic!("Invalid attribute type"),
            }
        } else {
            let mut input_line = String::new();
            println!("Enter {} for input {}", dim.dim_param(), dim.denotation());
            io::stdin().read_line(&mut input_line).unwrap_or_default();
            let x: i64 = input_line.trim().parse().unwrap_or_default();
            x
        }
    } else {
        panic!("Invalid dim")
    }
}

pub fn get_bytedata(name: &str) -> Vec<u8> {
    println!("Input filename for data {}: ", name);
    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap_or_default();

    let p = Path::new(input_line.trim());
    let fp = std::env::current_dir().unwrap().join(p);
    println!("PATH: {:?}", fp);
    std::fs::read(fp).unwrap()
}

pub fn get_bytedata_from_file(filename: &str) -> Vec<u8> {
    let p = Path::new(filename.trim());
    let fp = std::env::current_dir().unwrap().join(p);
    println!("PATH: {:?}", fp);
    std::fs::read(fp).unwrap()
}

pub fn make_external_inputs(
    graph: &onnx::GraphProto,
    inputs_file: &FileInputs,
) -> HashMap<String, Vec<u8>> {
    let mut map = HashMap::new();
    for input in graph.input.iter() {
        let input_name = input.name.as_ref().map_or(UNKNOWN, |v| v.as_str());
        if let Some(input_from_file) = inputs_file.inputs.iter().find(|i| i.name == input_name) {
            map.insert(
                input_name.to_string(),
                get_bytedata_from_file(&input_from_file.datafile),
            );
        } else {
            map.insert(input_name.to_string(), get_bytedata(input_name));
        }
    }
    map
}

pub fn make_inputs<'a>(
    graph: &'a onnx::GraphProto,
    external_data: &'a HashMap<String, Vec<u8>>,
    inputs_file: &FileInputs,
) -> HashMap<String, ArrayType> {
    let mut inputs: HashMap<String, ArrayType> = HashMap::new();
    for input in graph.input.iter() {
        use onnx::type_proto::Value;
        let input_name = input.name.as_ref().map_or(UNKNOWN, |v| v.as_str());
        match &input.type_.value {
            Some(Value::TensorType(t)) => match (t.elem_type, &t.shape) {
                (Some(et), protobuf::MessageField(Some(s))) => {
                    match make_tensor(
                        &s.dim
                            .iter()
                            .map(|d| {
                                get_dim(d, inputs_file.inputs.iter().find(|i| i.name == input_name))
                            })
                            .collect::<Vec<i64>>(),
                        external_data.get(input_name).unwrap(),
                        et,
                    ) {
                        Ok(a) => {
                            inputs.insert(input_name.to_string(), a);
                        }
                        Err(e) => {
                            println!(
                                "  Input: {} has tensor type {:?} but error: {}",
                                input_name, et, e
                            );
                        }
                    }
                }
                _ => {
                    panic!(
                        "  Input: {} has tensor type but no element type",
                        input_name
                    )
                }
            },
            Some(Value::SparseTensorType(_)) => {
                todo!("  Input: {} has type SparseTensor", input_name)
            },
            Some(Value::SequenceType(_)) => {
                todo!("  Input: {} has type Sequence", input_name)
            },
            Some(Value::MapType(_)) => {
                todo!("  Input: {} has type Map", input_name)
            },
            Some(Value::OptionalType(_)) => {
                todo!("  Input: {} has type Optional", input_name)
            },
            None => {
                panic!("  Input: {} has no type", input_name)
            }
        }
    }
    inputs
}

fn read_model_text(p: &Path) -> Result<onnx::ModelProto, Box<dyn Error>> {
    let file = std::fs::File::open(p)?;
    let mut reader = io::BufReader::new(file);
    let mut buf = String::new();
    reader.read_to_string(&mut buf)?;
    let model = protobuf::text_format::parse_from_str(&buf)?;
    Ok(model)
}
fn read_model_binary(p: &Path) -> Result<onnx::ModelProto, Box<dyn Error>> {
    let file = std::fs::File::open(p)?;
    let mut reader = io::BufReader::new(file);
    let model: onnx::ModelProto = protobuf::Message::parse_from_reader(&mut reader)?;
    Ok(model)
}

pub fn read_model(p: &Path) -> Result<onnx::ModelProto, Box<dyn Error>> {
    if let Ok(m) = read_model_binary(p) {
        Ok(m)
    } else {
        read_model_text(p)
    }
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
