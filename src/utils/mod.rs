use std::io::Read;
use std::os::raw::c_uchar;
use std::path;
use std::{collections::HashMap, error::Error, io, path::Path};

use ndarray::{ArrayD, IxDyn};
use protobuf::Enum;

use crate::onnx::tensor_proto::DataType;
use crate::onnx::{TensorProto, ValueInfoProto};
use crate::onnxparser::onnx;
use crate::FileInputs;
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

pub fn shape_safe_product<'a, B: 'a + std::iter::Product<&'a B> + std::default::Default, A: IntoIterator<Item = &'a B>>(shape: A) -> B {
    let mut piter = shape.into_iter().peekable();
    if piter.peek().is_none() {
        std::default::Default::default()
    } else {
        piter.product()
    }
}

impl ValueType {
    fn new(proto: onnx::tensor_proto::DataType) -> Result<ValueType, Box<dyn Error>> {
        match proto {
            DataType::UNDEFINED => Err("Undefined data type".into()),
            DataType::UINT8 => Ok(ValueType::U8),
            DataType::INT8 => Ok(ValueType::I8),
            DataType::UINT16 => Ok(ValueType::U16),
            DataType::INT16 => Ok(ValueType::I16),
            DataType::INT32 => Ok(ValueType::I32),
            DataType::INT64 => Ok(ValueType::I64),
            DataType::STRING => Ok(ValueType::String),
            DataType::BOOL => Ok(ValueType::Bool),
            DataType::FLOAT16 => Ok(ValueType::F16),
            DataType::DOUBLE => Ok(ValueType::F64),
            DataType::UINT32 => Ok(ValueType::U32),
            DataType::UINT64 => Ok(ValueType::U64),
            DataType::COMPLEX64 => Ok(ValueType::C64),
            DataType::COMPLEX128 => Ok(ValueType::C128),
            DataType::BFLOAT16 => Ok(ValueType::BF16),
            DataType::FLOAT
            | DataType::FLOAT8E4M3FN
            | DataType::FLOAT8E4M3FNUZ
            | DataType::FLOAT8E5M2
            | DataType::FLOAT8E5M2FNUZ => Ok(ValueType::F32),
        }
    }
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

pub trait ArrayTypeTrait: Clone + Copy + num::Zero {}

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
    pub fn ndim(&self) -> usize {
        match self {
            ArrayType::I8(a) => a.ndim(),
            ArrayType::I16(a) => a.ndim(),
            ArrayType::I32(a) => a.ndim(),
            ArrayType::I64(a) => a.ndim(),
            ArrayType::U8(a) => a.ndim(),
            ArrayType::U16(a) => a.ndim(),
            ArrayType::U32(a) => a.ndim(),
            ArrayType::U64(a) => a.ndim(),
            ArrayType::F16(a) => a.ndim(),
            ArrayType::BF16(a) => a.ndim(),
            ArrayType::F32(a) => a.ndim(),
            ArrayType::F64(a) => a.ndim(),
            ArrayType::C64(a) => a.ndim(),
            ArrayType::C128(a) => a.ndim(),
            ArrayType::Str(a) => a.ndim(),
            ArrayType::Bool(a) => a.ndim(),
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
    fn from_proto(proto: &ValueInfoProto) -> Result<Self, Box<dyn Error>> {
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

pub fn make_tensor_from_proto(proto: &TensorProto) -> Result<ArrayType, Box<dyn Error>> {
    let shape = &proto.dims;
    println!("  Tensor: {} data location: {:?}", proto.name(), proto.data_location());
    if let Some(DataType::STRING) = DataType::from_i32(proto.data_type()) {
        let bytedata = &proto.string_data;
        make_string_tensor(shape, bytedata)
    } else {
        let bytedata = proto.raw_data();
        make_tensor(shape, bytedata, proto.data_type())
    }
}

pub fn make_string_tensor(
    shape: &[i64],
    bytedata: &[impl AsRef<[u8]>],
) -> Result<ArrayType, Box<dyn Error>> {
    let shape = shape.iter().map(|v| *v as usize).collect::<Vec<usize>>();
    let a = ArrayD::<String>::from_shape_vec(
        IxDyn(&shape),
        bytedata
            .iter()
            .map(|v| String::from_utf8_lossy(v.as_ref()).to_string())
            .collect(),
    )?;
    Ok(ArrayType::Str(a))
}

pub fn make_tensor(
    shape: &[i64],
    bytedata: &[u8],
    data_type: i32,
) -> Result<ArrayType, Box<dyn Error>> {
    let enum_dt = DataType::from_i32(data_type).unwrap_or_default();
    let shape = shape.iter().map(|v| *v as usize).collect::<Vec<usize>>();
    match enum_dt {
        DataType::UNDEFINED => Err("Undefined data type".into()),
        DataType::INT8 => match bytemuck::try_cast_slice::<u8, i8>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<i8>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I8(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::INT16 => match bytemuck::try_cast_slice::<u8, i16>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<i16>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I16(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::INT32 => match bytemuck::try_cast_slice::<u8, i32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<i32>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I32(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::INT64 => match bytemuck::try_cast_slice::<u8, i64>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<i64>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::I64(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::UINT8 => match bytemuck::try_cast_slice::<u8, u8>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<u8>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U8(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::UINT16 => match bytemuck::try_cast_slice::<u8, u16>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<u16>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U16(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::UINT32 => match bytemuck::try_cast_slice::<u8, u32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<u32>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U32(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::UINT64 => match bytemuck::try_cast_slice::<u8, u64>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<u64>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::U64(a))
            }
            Err(e) => Err(e.to_string().into()),
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
            Err(e) => Err(e.to_string().into()),
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
            Err(e) => Err(e.to_string().into()),
        },
        DataType::DOUBLE => match bytemuck::try_cast_slice::<u8, f64>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape_safe_product(&shape));
                let a = ArrayD::<f64>::from_shape_vec(IxDyn(&shape), data.to_vec())?;
                Ok(ArrayType::F64(a))
            }
            Err(e) => Err(e.to_string().into()),
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
            Err(e) => Err(e.to_string().into()),
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
                if dlen != slen {
                    panic!("Data length {} does not match shape length {}", dlen, slen)
                }
                let a = if data.is_empty() {
                    ArrayD::<f32>::zeros(IxDyn(&shape))
                } else {
                    ArrayD::<f32>::from_shape_vec(IxDyn(&shape), data.to_vec())?
                };
                Ok(ArrayType::F32(a))
            }
            Err(e) => Err(e.to_string().into()),
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
            Err(e) => Err(e.to_string().into()),
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
    files: &[String],
) -> Result<HashMap<String, ArrayType>, Box<dyn Error>> {
    let mut map = HashMap::new();
    let mut external_inputs_map = HashMap::new();
    for input in files.iter() {
        let input_tensor = read_tensor(path::Path::new(&input))?;
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
            map.insert(
                input_name.to_string(),
                make_tensor_from_proto(input_from_file)?,
            );
        } else {
            return Err(format!("Output {} not found in inputs file", input_name).into());
        }
    }
    Ok(map)
}

fn make_output_tensors_from_files(
    graph: &onnx::GraphProto,
    files: &[String],
) -> Result<HashMap<String, ArrayType>, Box<dyn Error>> {
    let mut map = HashMap::new();
    let mut external_outputs_map = HashMap::new();
    for output in files.iter() {
        let ouput_tensor = read_tensor(path::Path::new(&output))?;
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
            return Err(format!("Output {} not found in inputs file", output_name).into());
        }
    }
    Ok(map)
}

pub fn make_external_inputs(
    graph: &onnx::GraphProto,
    fileinputs: &FileInputs,
) -> Result<HashMap<String, ArrayType>, Box<dyn Error>> {
    if fileinputs.inputs.is_empty() {
        return Ok(HashMap::new());
    }
    make_input_tensors_from_files(graph, &fileinputs.inputs)
}

pub fn make_external_outputs(
    graph: &onnx::GraphProto,
    fileinputs: &FileInputs,
) -> Result<HashMap<String, ArrayType>, Box<dyn Error>> {
    if fileinputs.outputs.is_empty() {
        return Ok(HashMap::new());
    }
    make_output_tensors_from_files(graph, &fileinputs.outputs)
}

pub fn make_graph_outputs(
    graph: &onnx::GraphProto,
) -> Result<HashMap<String, OutputInfo>, Box<dyn Error>> {
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
    let merr = read_model_binary(p);
    match merr {
        Ok(m) => Ok(m),
        Err(e) => {
            eprintln!("Error reading binary model: {}", e);
            read_model_text(p)
        }
    }
}

pub fn read_tensor(p: &Path) -> Result<onnx::TensorProto, Box<dyn Error>> {
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
