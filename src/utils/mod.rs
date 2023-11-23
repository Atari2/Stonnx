use anyhow::anyhow;
use std::io::Read;
use std::os::raw::c_uchar;
use std::path::PathBuf;
use std::{collections::HashMap, io, path::Path};

use ndarray::{ArrayD, IxDyn};
use protobuf::Enum;

use crate::onnx::tensor_proto::DataType;
use crate::onnx::{NodeProto, TensorProto, ValueInfoProto};
use crate::onnxparser::onnx;
use crate::FileInputs;
use half::{bf16, f16};
use once_cell::sync::OnceCell;
use num::complex::Complex;

type Complex64 = Complex<f32>;
type Complex128 = Complex<f64>;

static UNKNOWN: &str = "<unknown>";

pub type BoxResult<A> = anyhow::Result<A>;

pub static VERBOSE: OnceCell<usize> = OnceCell::new();

pub struct NDIndex<'a> {
    indices: &'a [usize],
    current_index: Vec<usize>,
    indices_empty_flag: bool,
}

impl<'a> NDIndex<'a> {
    pub fn new(shape: &'a [usize]) -> Self {
        Self {
            indices: shape,
            current_index: if shape.iter().all(|v| *v != 0) {
                vec![0; shape.len()]
            } else {
                vec![]
            },
            indices_empty_flag: shape.is_empty(),
        }
    }
}

impl Iterator for NDIndex<'_> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.indices_empty_flag {
            self.indices_empty_flag = false;
            return Some(vec![]);
        }
        if self.current_index.is_empty() {
            return None;
        }
        let result = self.current_index.clone();
        let mut i = self.current_index.len() - 1;
        loop {
            if self.current_index[i] < self.indices[i] - 1 {
                self.current_index[i] += 1;
                break;
            } else {
                self.current_index[i] = 0;
                if i == 0 {
                    self.current_index = vec![];
                    break;
                }
                i -= 1;
            }
        }
        Some(result)
    }
}

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

#[derive(Debug)]
pub enum OperationResult {
    Single(ArrayType),
    OptionalDouble((ArrayType, Option<ArrayType>)),
    Double((ArrayType, ArrayType)),
    Multiple(Vec<ArrayType>),
}

impl From<ArrayType> for OperationResult {
    fn from(r: ArrayType) -> Self {
        OperationResult::Single(r)
    }
}

impl From<(ArrayType, Option<ArrayType>)> for OperationResult {
    fn from(r: (ArrayType, Option<ArrayType>)) -> Self {
        OperationResult::OptionalDouble(r)
    }
}

impl From<(ArrayType, ArrayType)> for OperationResult {
    fn from(r: (ArrayType, ArrayType)) -> Self {
        OperationResult::Double(r)
    }
}

impl From<Vec<ArrayType>> for OperationResult {
    fn from(r: Vec<ArrayType>) -> Self {
        OperationResult::Multiple(r)
    }
}

impl OperationResult {}

pub type OperationFn = for<'a, 'b, 'c> fn(
    &'a [&'b ArrayType],
    &'c NodeProto,
    i64,
    usize,
) -> BoxResult<OperationResult>;

pub fn shape_safe_product<
    'a,
    B: 'a + std::iter::Product<&'a B> + std::default::Default,
    A: IntoIterator<Item = &'a B>,
>(
    shape: A,
) -> B {
    let mut piter = shape.into_iter().peekable();
    if piter.peek().is_none() {
        std::default::Default::default()
    } else {
        piter.product()
    }
}

impl ValueType {
    fn new(proto: onnx::tensor_proto::DataType) -> BoxResult<ValueType> {
        match proto {
            DataType::UNDEFINED => Err(anyhow!("Undefined data type")),
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

macro_rules! impl_into_array_type {
    ($t:ty, $v:ident) => {
        impl From<ArrayD<$t>> for ArrayType {
            fn from(a: ArrayD<$t>) -> Self {
                ArrayType::$v(a)
            }
        }
        impl<'a> TryFrom<&'a ArrayType> for &'a ArrayD<$t> {
            type Error = anyhow::Error;
            fn try_from(a: &'a ArrayType) -> BoxResult<&'a ArrayD<$t>> {
                match a {
                    ArrayType::$v(a) => Ok(&a),
                    _ => Err(anyhow!("Wrong type")),
                }
            }
        }
    };
    () => {
        compile_error!("impl_into_array_type!() requires a type argument");
    };
}

impl_into_array_type!(i8, I8);
impl_into_array_type!(i16, I16);
impl_into_array_type!(i32, I32);
impl_into_array_type!(i64, I64);
impl_into_array_type!(u8, U8);
impl_into_array_type!(u16, U16);
impl_into_array_type!(u32, U32);
impl_into_array_type!(u64, U64);
impl_into_array_type!(f16, F16);
impl_into_array_type!(bf16, BF16);
impl_into_array_type!(f32, F32);
impl_into_array_type!(f64, F64);
impl_into_array_type!(Complex64, C64);
impl_into_array_type!(Complex128, C128);
impl_into_array_type!(String, Str);
impl_into_array_type!(bool, Bool);

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
    pub fn to_file(&self, dir: &Path, name: &str) -> BoxResult<()> {
        let name = name.replace('/', "_");
        use ndarray_npy::write_npy;
        match self {
            ArrayType::I8(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::I16(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::I32(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::I64(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::U8(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::U16(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::U32(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::U64(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::F16(_) => todo!("Half precision data type not supported"),
            ArrayType::BF16(_) => todo!("BFloat16 data type not supported"),
            ArrayType::F32(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::F64(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::C64(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::C128(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            ArrayType::Str(_) => todo!("String data type not supported"),
            ArrayType::Bool(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
        }
    }
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
    pub fn data_type(&self) -> onnx::tensor_proto::DataType {
        // FIXME: types such as FLOAT8E4M3FN all map to FLOAT right now
        //        need to handle them properly
        match self {
            ArrayType::I64(_) => DataType::INT64,
            ArrayType::F32(_) => DataType::FLOAT,
            ArrayType::I8(_) => DataType::INT8,
            ArrayType::I16(_) => DataType::INT16,
            ArrayType::I32(_) => DataType::INT32,
            ArrayType::U8(_) => DataType::UINT8,
            ArrayType::U16(_) => DataType::UINT16,
            ArrayType::U32(_) => DataType::UINT32,
            ArrayType::U64(_) => DataType::UINT64,
            ArrayType::F16(_) => DataType::FLOAT16,
            ArrayType::BF16(_) => DataType::BFLOAT16,
            ArrayType::F64(_) => DataType::DOUBLE,
            ArrayType::C64(_) => DataType::COMPLEX64,
            ArrayType::C128(_) => DataType::COMPLEX128,
            ArrayType::Str(_) => DataType::STRING,
            ArrayType::Bool(_) => DataType::BOOL,
        }
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
            ndarray_npy::write_npy(format!("{}_intermediate_outputs/{}_{}.npy", operation, COUNTER, name), a)?;
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
    ($name:ident) => {
        {
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
        }
    };
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
    if let Some(DataType::STRING) = DataType::from_i32(proto.data_type()) {
        let bytedata = &proto.string_data;
        make_string_tensor(shape, bytedata)
    } else {
        make_tensor(shape, proto, proto.data_type())
    }
}

pub fn make_string_tensor(shape: &[i64], bytedata: &[impl AsRef<[u8]>]) -> BoxResult<ArrayType> {
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

pub fn make_tensor(shape: &[i64], proto: &TensorProto, data_type: i32) -> BoxResult<ArrayType> {
    let enum_dt = DataType::from_i32(data_type).unwrap_or_default();
    let shape = shape.iter().map(|v| *v as usize).collect::<Vec<usize>>();
    let bytedata = proto.raw_data();
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
        DataType::INT32 => {
            let data = if let Some(data) = &proto.raw_data {
                match bytemuck::try_cast_slice::<u8, i32>(data) {
                    Ok(data) => data,
                    Err(e) => return Err(anyhow!(e)),
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
                match bytemuck::try_cast_slice::<u8, i64>(data) {
                    Ok(data) => data,
                    Err(e) => return Err(anyhow!(e)),
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
        DataType::FLOAT8E4M3FN
        | DataType::FLOAT8E4M3FNUZ
        | DataType::FLOAT8E5M2FNUZ
        | DataType::FLOAT8E5M2 => {
            todo!("Data type {:?} not supported", enum_dt);
        }
        DataType::FLOAT => {
            let data = if let Some(data) = &proto.raw_data {
                match bytemuck::try_cast_slice::<u8, f32>(data) {
                    Ok(data) => data,
                    Err(e) => return Err(anyhow!(e)),
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
