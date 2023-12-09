use std::ops::AddAssign;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use clap::Parser;
use half::{bf16, f16};
use ndarray::ArrayD;
use ndarray::ScalarOperand;
use num::traits::AsPrimitive;
use num::Complex;
use once_cell::sync::OnceCell;
use serde::Deserialize;
use serde::Serialize;

use crate::onnx::{self, tensor_proto::DataType, NodeProto};

pub type Complex64 = Complex<f32>;
pub type Complex128 = Complex<f64>;

pub static UNKNOWN: &str = "<unknown>";

pub type BoxResult<A> = anyhow::Result<A>;

pub static VERBOSE: OnceCell<VerbosityLevel> = OnceCell::new();

#[derive(Parser, Debug)]
/// Parse and execute inference on pre-trained ONNX models
pub struct Args {
    /// Path to the folder containing the model to run
    ///
    /// This folder should contain the model's `inputs.json`
    ///
    /// This file should contain a "modelpath" property which indicates the path to the model's `.onnx` file
    /// if this path is relative, it will be relative to the model's folder
    ///
    /// This file should also contain an "inputs" property which is an array of paths to the model's inputs
    /// if these paths are relative, they will be relative to the model's folder
    ///
    /// This file should also contain an "outputs" property which is an array of paths to the model's expected outputs
    /// if these paths are relative, they will be relative to the model's folder, these outputs will be compared to the outputs of the model
    #[arg(short, long)]
    pub model: PathBuf,

    /// Set verbosity level
    ///
    /// 0 - No output except basic logging
    ///
    /// 2 - Output all results from operators into .npy files
    ///
    /// 4 - Output intermediate results from operators into .npy files (only supported by conv for now)
    #[arg(short, long, default_value = "0")]
    pub verbose: u64,

    /// Generate json file representing the graph of the model
    ///
    /// This JSON file is meant to be parsed and printed by the C# program `ONNXGraphLayout`
    #[arg(short, long, default_value = "false")]
    pub gengraph: bool,

    /// Fail immediately if an operator is not implemented yet, otherwise continue and execute the model until panic
    #[arg(short, long, default_value = "false")]
    pub failfast: bool,
}

#[derive(Serialize, Deserialize)]
pub struct FileInputs {
    pub inputs: Vec<PathBuf>,
    pub outputs: Vec<PathBuf>,
    pub modelpath: PathBuf,
}

impl FileInputs {
    pub fn extend_paths(&mut self, modelname: &Path) {
        self.inputs = self
            .inputs
            .iter()
            .map(|s| {
                if s.is_absolute() {
                    s.clone()
                } else {
                    Path::new("models").join(modelname).join(s)
                }
            })
            .collect();
        self.outputs = self
            .outputs
            .iter()
            .map(|s| {
                if s.is_absolute() {
                    s.clone()
                } else {
                    Path::new("models").join(modelname).join(s)
                }
            })
            .collect();
        self.modelpath = if self.modelpath.is_relative() {
            Path::new("models")
                .join(modelname)
                .join(self.modelpath.clone())
        } else {
            self.modelpath.clone()
        };
    }
}

pub const MAX_OPSET_VERSION: i64 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VerbosityLevel {
    None,
    Informational,
    Results,
    Intermediate,
}

impl VerbosityLevel {
    pub fn new(level: usize) -> Self {
        match level {
            0 => VerbosityLevel::None,
            1 => VerbosityLevel::Informational,
            2 | 3 => VerbosityLevel::Results,
            4.. => VerbosityLevel::Intermediate,
            _ => unreachable!(),
        }
    }
}

#[macro_export]
macro_rules! print_at_level {
    ($level:expr, $($arg:tt)*) => {
        if $crate::VERBOSE.get() >= Some(&$level) {
            println!($($arg)*);
        }
    };
}

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

impl ValueType {
    pub fn new(proto: onnx::tensor_proto::DataType) -> BoxResult<ValueType> {
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

/// Sparse array type, not supported yet
#[derive(Debug, Clone)]
pub enum SparseArrayType {}

/// Map type, not supported yet
#[derive(Debug, Clone)]
pub enum MapType {}

/// Sequence type, not supported yet
#[derive(Debug, Clone)]
pub enum SequenceType {}

/// Optional type, not supported yet
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum OptionalType {
    Array(ArrayType),
    SparseArray(SparseArrayType),
    Map(MapType),
    Sequence(SequenceType),
    None,
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

pub trait ArrayElement:
    Copy
    + Clone
    + ndarray_npy::WritableElement
    + num::Num
    + num::FromPrimitive
    + Default
    + PartialOrd
    + AddAssign
    + ScalarOperand
    + IsNan
    + MinMax
    + std::iter::Sum
    + HasSqrt
    + 'static
{
}

impl<T> ArrayElement for T where
    T: Copy
        + Clone
        + ndarray_npy::WritableElement
        + num::Num
        + num::FromPrimitive
        + Default
        + PartialOrd
        + AddAssign
        + ScalarOperand
        + IsNan
        + MinMax
        + std::iter::Sum
        + HasSqrt
        + 'static
{
}

pub trait F32IntoType<T> {
    fn as_(self) -> T;
}

pub trait IsNan {
    #[allow(clippy::wrong_self_convention)]
    fn is_nan(self) -> bool;
}

pub trait MinMax {
    const MAX: Self;
    const MIN: Self;
    fn max(self, rhs: Self) -> Self;
    fn min(self, lhs: Self) -> Self;
}
pub trait HasSqrt {
    fn sqrt(self) -> Self;
}

impl IsNan for f32 {
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}
impl IsNan for f64 {
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}

impl F32IntoType<Complex64> for f32 {
    fn as_(self) -> Complex64 {
        Complex64::new(self, self)
    }
}

impl F32IntoType<Complex128> for f32 {
    fn as_(self) -> Complex128 {
        Complex128::new(self as f64, self as f64)
    }
}

impl F32IntoType<f16> for f32 {
    fn as_(self) -> f16 {
        f16::from_f32(self)
    }
}

impl F32IntoType<bf16> for f32 {
    fn as_(self) -> bf16 {
        bf16::from_f32(self)
    }
}

impl F32IntoType<String> for f32 {
    fn as_(self) -> String {
        self.to_string()
    }
}

impl F32IntoType<bool> for f32 {
    fn as_(self) -> bool {
        self != 0.0
    }
}

macro_rules! impl_from_f32 {
    ($t:ty) => {
        impl F32IntoType<$t> for f32
        where
            f32: AsPrimitive<$t>,
        {
            fn as_(self) -> $t {
                num::traits::AsPrimitive::as_(self)
            }
        }
    };
}

macro_rules! impl_is_nan {
    ($t:ty) => {
        impl IsNan for $t {
            fn is_nan(self) -> bool {
                false
            }
        }
    };
}

macro_rules! impl_minmax {
    ($t:ty) => {
        impl MinMax for $t {
            const MAX: Self = <$t>::MAX;
            const MIN: Self = <$t>::MIN;
            fn min(self, rhs: Self) -> Self {
                if self < rhs {
                    self
                } else {
                    rhs
                }
            }
            fn max(self, rhs: Self) -> Self {
                if self > rhs {
                    self
                } else {
                    rhs
                }
            }
        }
    };
}

macro_rules! impl_has_sqrt_i {
    ($t:ty) => {
        impl HasSqrt for $t {
            fn sqrt(self) -> Self {
                num::integer::Roots::sqrt(&self)
            }
        }
    };
}

macro_rules! impl_has_sqrt_f {
    ($t:ty) => {
        impl HasSqrt for $t {
            fn sqrt(self) -> Self {
                num::Float::sqrt(self)
            }
        }
    };
}

impl_from_f32!(i8);
impl_from_f32!(i16);
impl_from_f32!(i32);
impl_from_f32!(i64);
impl_from_f32!(u8);
impl_from_f32!(u16);
impl_from_f32!(u32);
impl_from_f32!(u64);
impl_from_f32!(f32);
impl_from_f32!(f64);

impl_is_nan!(i8);
impl_is_nan!(i16);
impl_is_nan!(i32);
impl_is_nan!(i64);
impl_is_nan!(u8);
impl_is_nan!(u16);
impl_is_nan!(u32);
impl_is_nan!(u64);

impl_minmax!(i8);
impl_minmax!(i16);
impl_minmax!(i32);
impl_minmax!(i64);
impl_minmax!(u8);
impl_minmax!(u16);
impl_minmax!(u32);
impl_minmax!(u64);
impl_minmax!(f32);
impl_minmax!(f64);

impl_has_sqrt_i!(i8);
impl_has_sqrt_i!(i16);
impl_has_sqrt_i!(i32);
impl_has_sqrt_i!(i64);
impl_has_sqrt_i!(u8);
impl_has_sqrt_i!(u16);
impl_has_sqrt_i!(u32);
impl_has_sqrt_i!(u64);
impl_has_sqrt_f!(f32);
impl_has_sqrt_f!(f64);

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
pub struct Complex64Repr {
    pub _val: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
pub struct Complex128Repr {
    pub _val: [f64; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
pub struct HalfFloat {
    pub _val: u16,
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
