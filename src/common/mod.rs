use std::ops::AddAssign;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;

use anyhow::anyhow;
use clap::Parser;
use half::{bf16, f16};
use ndarray::ArrayD;
use ndarray::ScalarOperand;
use num::traits::AsPrimitive;
use num::Complex;
use serde::Deserialize;
use serde::Serialize;

use crate::onnx::{self, tensor_proto::DataType, NodeProto};

pub type Complex64 = Complex<f32>;
pub type Complex128 = Complex<f64>;

/// Static string that is used when the name of a node is not known
pub static UNKNOWN: &str = "<unknown>";

pub type BoxResult<A> = anyhow::Result<A>;

/// This static variable holds the verbosity level of the program
pub static VERBOSE: AtomicUsize = AtomicUsize::new(VerbosityLevel::Minimal as usize);

#[derive(Parser, Debug)]
/// Parse and execute inference on pre-trained ONNX models
pub struct Args {
    /// Path to the folder containing the model to run
    ///
    /// If this path is relative, it will be assumed to be relative to $cwd/models, where $cwd is the current working directory
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
    /// -1 - No output at all
    ///
    /// 0 - No output except basic logging
    ///
    /// 1 - Outputs information about each operator that is executed
    ///
    /// 2 - Output all results from operators into .npy files
    ///
    /// 4 - Output intermediate results from operators into .npy files (only supported by conv for now)
    #[arg(short, long, default_value = "0", allow_hyphen_values = true)]
    pub verbose: i64,

    /// Generate json/dot file representing the graph of the model
    ///
    /// This JSON file is meant to be parsed and printed by the C# program `ONNXGraphLayout`
    /// The dot file can be converted to an image using the `dot` program from graphviz
    #[arg(short, long, default_value = "false")]
    pub gengraph: bool,

    /// Type of graph to generate, either json or dot, only used if `gengraph` is true
    ///
    /// Default is json
    #[arg(long, default_value = "json")]
    pub graphtype: String,

    /// Fail immediately if an operator is not implemented yet, otherwise continue and execute the model until panic
    #[arg(short, long, default_value = "false")]
    pub failfast: bool,
}

impl Args {
    #[allow(dead_code)]
    /// Only used in the library version of the program
    pub fn from_parts(
        model: PathBuf,
        verbose: i64,
        gengraph: bool,
        graphtype: String,
        failfast: bool,
    ) -> Self {
        Self {
            model,
            verbose,
            gengraph,
            graphtype,
            failfast,
        }
    }

    #[allow(dead_code)]
    /// Only used in the library version of the program
    pub fn new(
        model: PathBuf,
        verbose: VerbosityLevel,
        gengraph: bool,
        graphtype: String,
        failfast: bool,
    ) -> Self {
        Self {
            model,
            verbose: verbose as i64,
            gengraph,
            graphtype,
            failfast,
        }
    }
}

#[derive(Serialize, Deserialize)]
/// This struct holds the names of the inputs and outputs of a model
/// as well as the path to the model's `.onnx` file
pub struct FileInputs {
    pub inputs: Vec<PathBuf>,
    pub outputs: Vec<PathBuf>,
    pub modelpath: PathBuf,
}

impl FileInputs {
    /// This function extends the paths of the inputs and outputs of a model
    ///
    /// If the paths are relative, they will be relative to the model's folder
    ///
    /// If the paths are absolute, they are untouched
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

/// Maximum supported opset version
pub const MAX_OPSET_VERSION: i64 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
/// Indicates the verbosity level of the program
pub enum VerbosityLevel {
    Silent = 0,
    /// No output except basic logging
    Minimal = 1,
    /// Outputs information about each operator that is executed
    Informational = 2,
    /// Output all results from operators into .npy files
    Results = 3,
    /// Output intermediate results from operators into .npy files (only supported by conv for now)
    Intermediate = 4,
}

impl VerbosityLevel {
    pub fn new(level: i64) -> Self {
        match level {
            -1 => VerbosityLevel::Silent,
            0 => VerbosityLevel::Minimal,
            1 => VerbosityLevel::Informational,
            2 | 3 => VerbosityLevel::Results,
            4.. => VerbosityLevel::Intermediate,
            _ => unreachable!(),
        }
    }
}

impl std::cmp::PartialEq<usize> for VerbosityLevel {
    fn eq(&self, other: &usize) -> bool {
        *self as usize == *other
    }
}

impl std::cmp::PartialOrd<usize> for VerbosityLevel {
    fn partial_cmp(&self, other: &usize) -> Option<std::cmp::Ordering> {
        (*self as usize).partial_cmp(other)
    }
}

impl std::cmp::PartialEq<VerbosityLevel> for usize {
    fn eq(&self, other: &VerbosityLevel) -> bool {
        *self == *other as usize
    }
}

impl std::cmp::PartialOrd<VerbosityLevel> for usize {
    fn partial_cmp(&self, other: &VerbosityLevel) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&(*other as usize))
    }
}

#[macro_export]
/// This macro prints a message if the verbosity level is high enough
macro_rules! print_at_level {
    ($level:expr, $($arg:tt)*) => {
        if $crate::common::VERBOSE.load(std::sync::atomic::Ordering::Relaxed) >= $level {
            println!($($arg)*);
        }
    };
}

/// This is a helper struct that creates an iterator over the indices of a multidimensional array
///
/// This is used just like python's `numpy.ndindex`
///
/// The shape is visited in row-major order
/// ```
/// use stonnx_api::common::NDIndex;
///
/// let shape = vec![2, 3, 4];
/// let mut ndindex = NDIndex::new(&shape);
/// for i in ndindex {
///     println!("{:?}", i); // [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 0], [0, 1, 1], ...
/// }
/// ```
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
/// This enum represents the type of a TensorType
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
/// This structs encapsulates the result of an operator
pub struct OperatorResult {
    pub result: Vec<TensorType>,
}

impl From<TensorType> for OperatorResult {
    fn from(r: TensorType) -> Self {
        OperatorResult { result: vec![r] }
    }
}

impl From<(TensorType, Option<TensorType>)> for OperatorResult {
    fn from(r: (TensorType, Option<TensorType>)) -> Self {
        let v = if let (r, Some(l)) = r {
            vec![r, l]
        } else {
            vec![r.0]
        };
        OperatorResult { result: v }
    }
}

impl From<(TensorType, TensorType)> for OperatorResult {
    fn from(r: (TensorType, TensorType)) -> Self {
        OperatorResult {
            result: vec![r.0, r.1],
        }
    }
}

impl From<Vec<TensorType>> for OperatorResult {
    fn from(r: Vec<TensorType>) -> Self {
        OperatorResult { result: r }
    }
}

/// This is the type of an operator function
///
/// It takes a slice of inputs, the node that is being executed, the opset version, and the number of outputs
///
/// It returns a `BoxResult` containing the result of the operation or an error
pub type OperationFn = for<'a, 'b, 'c> fn(
    &'a [&'b TensorType],
    &'c NodeProto,
    i64,
    usize,
) -> BoxResult<OperatorResult>;

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
/// Represents an ONNX tensor
pub enum TensorType {
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
pub enum SparseTensorType {}

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
    Tensor(TensorType),
    SparseTensor(SparseTensorType),
    Map(MapType),
    Sequence(SequenceType),
    None,
}

/// This macro creates a function that implements the `From<ArrayD<A>>` trait for TensorType
/// and the `TryFrom<&TensorType>` trait for `&ArrayD<A>`
///
/// This macro is used to implement the `From` and `TryFrom` traits for all the types that are supported by ndarray  
macro_rules! impl_into_array_type {
    ($t:ty, $v:ident) => {
        impl From<ArrayD<$t>> for TensorType {
            fn from(a: ArrayD<$t>) -> Self {
                TensorType::$v(a)
            }
        }
        impl<'a> TryFrom<&'a TensorType> for &'a ArrayD<$t> {
            type Error = anyhow::Error;
            fn try_from(a: &'a TensorType) -> BoxResult<&'a ArrayD<$t>> {
                match a {
                    TensorType::$v(a) => Ok(&a),
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

/// This is the trait that represents the types that are supported by TensorType
///
/// Most of the time this is the only trait needed to make a function generic over the types supported by TensorType
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
    + std::marker::Sync
    + std::marker::Send
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
        + std::marker::Sync
        + std::marker::Send
        + 'static
{
}

/// This trait implements the `as_` method that converts from &lt;T&gt; from f32
pub trait F32IntoType<T> {
    fn as_(self) -> T;
}

/// This trait is a is_nan trait that is implemented for all types supported by TensorType
/// including integers and Complex numbers
pub trait IsNan {
    #[allow(clippy::wrong_self_convention)]
    fn is_nan(self) -> bool;
}

/// This trait is a min and max trait that is implemented for all types supported by TensorType
pub trait MinMax {
    const MAX: Self;
    const MIN: Self;
    fn max(self, rhs: Self) -> Self;
    fn min(self, lhs: Self) -> Self;
}

/// This trait is a sqrt trait that is implemented for all types supported by TensorType
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

/// This macro implements the `F32IntoType` trait for all the types supported by TensorType
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

/// This macro implements the `IsNan` trait for all the types supported by TensorType
macro_rules! impl_is_nan {
    ($t:ty) => {
        impl IsNan for $t {
            fn is_nan(self) -> bool {
                false
            }
        }
    };
}

/// This macro implements the `MinMax` trait for all the types supported by TensorType
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

/// This macro implements the `HasSqrt` trait for all the integer types supported by TensorType
macro_rules! impl_has_sqrt_i {
    ($t:ty) => {
        impl HasSqrt for $t {
            fn sqrt(self) -> Self {
                num::integer::Roots::sqrt(&self)
            }
        }
    };
}

/// This macro implements the `HasSqrt` trait for all the float types supported by TensorType
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

/// This struct is used to represent a complex 64-bit number in numpy, it is purely for the sake of reading from raw data arrays in models
/// ```c
/// typedef struct
/// {
///     float _Val[2];
/// } npy_cfloat;
///
/// #define NPY_COMPLEX64 NPY_CFLOAT
///         typedef float npy_float32;
///         typedef npy_cfloat npy_complex64;
/// ```

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
pub struct Complex64Repr {
    pub _val: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
/// This struct is used to represent a complex 128-bit number in numpy, it is purely for the sake of reading from raw data arrays in models
/// ```c
/// typedef struct
/// {
///     double _Val[2];
/// } npy_cdouble;
///
/// #define NPY_COMPLEX128 NPY_CDOUBLE
///         typedef double npy_float64;
///         typedef npy_cdouble npy_complex128;
/// ```
pub struct Complex128Repr {
    pub _val: [f64; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
/// This struct is used to represent a half precision number in numpy, it is purely for the sake of reading from raw data arrays in models
pub struct HalfFloat {
    pub _val: u16,
}

impl std::fmt::Display for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorType::I8(_) => write!(f, "I8"),
            TensorType::I16(_) => write!(f, "I16"),
            TensorType::I32(_) => write!(f, "I32"),
            TensorType::U8(_) => write!(f, "U8"),
            TensorType::U16(_) => write!(f, "U16"),
            TensorType::U32(_) => write!(f, "U32"),
            TensorType::I64(_) => write!(f, "I64"),
            TensorType::U64(_) => write!(f, "U64"),
            TensorType::F16(_) => write!(f, "F16"),
            TensorType::BF16(_) => write!(f, "BF16"),
            TensorType::F32(_) => write!(f, "F32"),
            TensorType::F64(_) => write!(f, "F64"),
            TensorType::C64(_) => write!(f, "C64"),
            TensorType::C128(_) => write!(f, "C128"),
            TensorType::Str(_) => write!(f, "STR"),
            TensorType::Bool(_) => write!(f, "BOOL"),
        }
    }
}

impl TensorType {
    /// Writes the data of a TensorType to a file
    ///
    /// The file will be written in the numpy format, using the directory and name provided
    pub fn to_file(&self, dir: &Path, name: &str) -> BoxResult<()> {
        let name = name.replace('/', "_");
        use ndarray_npy::write_npy;
        match self {
            TensorType::I8(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::I16(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::I32(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::I64(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::U8(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::U16(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::U32(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::U64(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::F16(_) => todo!("Half precision data type not supported"),
            TensorType::BF16(_) => todo!("BFloat16 data type not supported"),
            TensorType::F32(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::F64(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::C64(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::C128(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
            TensorType::Str(_) => todo!("String data type not supported"),
            TensorType::Bool(data) => Ok(write_npy(dir.join(name).with_extension("npy"), data)?),
        }
    }

    /// Returns the shape of a TensorType
    pub fn shape(&self) -> &[usize] {
        match self {
            TensorType::I8(a) => a.shape(),
            TensorType::I16(a) => a.shape(),
            TensorType::I32(a) => a.shape(),
            TensorType::I64(a) => a.shape(),
            TensorType::U8(a) => a.shape(),
            TensorType::U16(a) => a.shape(),
            TensorType::U32(a) => a.shape(),
            TensorType::U64(a) => a.shape(),
            TensorType::F16(a) => a.shape(),
            TensorType::BF16(a) => a.shape(),
            TensorType::F32(a) => a.shape(),
            TensorType::F64(a) => a.shape(),
            TensorType::C64(a) => a.shape(),
            TensorType::C128(a) => a.shape(),
            TensorType::Str(a) => a.shape(),
            TensorType::Bool(a) => a.shape(),
        }
    }

    /// Returns the number of dimensions of a TensorType
    pub fn ndim(&self) -> usize {
        match self {
            TensorType::I8(a) => a.ndim(),
            TensorType::I16(a) => a.ndim(),
            TensorType::I32(a) => a.ndim(),
            TensorType::I64(a) => a.ndim(),
            TensorType::U8(a) => a.ndim(),
            TensorType::U16(a) => a.ndim(),
            TensorType::U32(a) => a.ndim(),
            TensorType::U64(a) => a.ndim(),
            TensorType::F16(a) => a.ndim(),
            TensorType::BF16(a) => a.ndim(),
            TensorType::F32(a) => a.ndim(),
            TensorType::F64(a) => a.ndim(),
            TensorType::C64(a) => a.ndim(),
            TensorType::C128(a) => a.ndim(),
            TensorType::Str(a) => a.ndim(),
            TensorType::Bool(a) => a.ndim(),
        }
    }

    /// Returns the data type of a TensorType, useful when you want to know the type of a tensor without using a match statement
    pub fn value_type(&self) -> ValueType {
        match self {
            TensorType::I64(_) => ValueType::I64,
            TensorType::F32(_) => ValueType::F32,
            TensorType::I8(_) => ValueType::I8,
            TensorType::I16(_) => ValueType::I16,
            TensorType::I32(_) => ValueType::I32,
            TensorType::U8(_) => ValueType::U8,
            TensorType::U16(_) => ValueType::U16,
            TensorType::U32(_) => ValueType::U32,
            TensorType::U64(_) => ValueType::U64,
            TensorType::F16(_) => ValueType::F16,
            TensorType::BF16(_) => ValueType::BF16,
            TensorType::F64(_) => ValueType::F64,
            TensorType::C64(_) => ValueType::C64,
            TensorType::C128(_) => ValueType::C128,
            TensorType::Str(_) => ValueType::String,
            TensorType::Bool(_) => ValueType::Bool,
        }
    }

    /// Returns the *internal* data type of a TensorType, this is useful when you have a onnx::tensor_proto::DataType instead of a ValueType
    pub fn data_type(&self) -> onnx::tensor_proto::DataType {
        // FIXME: types such as FLOAT8E4M3FN all map to FLOAT right now
        //        need to handle them properly
        match self {
            TensorType::I64(_) => DataType::INT64,
            TensorType::F32(_) => DataType::FLOAT,
            TensorType::I8(_) => DataType::INT8,
            TensorType::I16(_) => DataType::INT16,
            TensorType::I32(_) => DataType::INT32,
            TensorType::U8(_) => DataType::UINT8,
            TensorType::U16(_) => DataType::UINT16,
            TensorType::U32(_) => DataType::UINT32,
            TensorType::U64(_) => DataType::UINT64,
            TensorType::F16(_) => DataType::FLOAT16,
            TensorType::BF16(_) => DataType::BFLOAT16,
            TensorType::F64(_) => DataType::DOUBLE,
            TensorType::C64(_) => DataType::COMPLEX64,
            TensorType::C128(_) => DataType::COMPLEX128,
            TensorType::Str(_) => DataType::STRING,
            TensorType::Bool(_) => DataType::BOOL,
        }
    }

    /// Compares two TensorTypes for equality, implementing numpy's `allclose` function
    pub fn allclose(
        &self,
        other: &Self,
        rtol: Option<f32>,
        atol: Option<f32>,
        equal_nan: Option<bool>,
    ) -> bool {
        match (self, other) {
            (TensorType::I8(a), TensorType::I8(b)) => allclose(a, b, rtol, atol, equal_nan),
            (TensorType::I16(a), TensorType::I16(b)) => allclose(a, b, rtol, atol, equal_nan),
            (TensorType::I32(a), TensorType::I32(b)) => allclose(a, b, rtol, atol, equal_nan),
            (TensorType::I64(a), TensorType::I64(b)) => allclose(a, b, rtol, atol, equal_nan),
            (TensorType::U8(a), TensorType::U8(b)) => {
                allclose_unsigned(a, b, rtol, atol, equal_nan)
            }
            (TensorType::U16(a), TensorType::U16(b)) => {
                allclose_unsigned(a, b, rtol, atol, equal_nan)
            }
            (TensorType::U32(a), TensorType::U32(b)) => {
                allclose_unsigned(a, b, rtol, atol, equal_nan)
            }
            (TensorType::U64(a), TensorType::U64(b)) => {
                allclose_unsigned(a, b, rtol, atol, equal_nan)
            }
            (TensorType::F16(_), TensorType::F16(_)) => {
                todo!("Half precision data type not supported")
            }
            (TensorType::BF16(_), TensorType::BF16(_)) => {
                todo!("BFloat16 data type not supported")
            }
            (TensorType::F32(a), TensorType::F32(b)) => allclose(a, b, rtol, atol, equal_nan),
            (TensorType::F64(a), TensorType::F64(b)) => allclose(a, b, rtol, atol, equal_nan),
            (TensorType::C64(_), TensorType::C64(_)) => {
                todo!("Complex64 data type not supported")
            }
            (TensorType::C128(_), TensorType::C128(_)) => {
                todo!("Complex128 data type not supported")
            }
            (TensorType::Str(a), TensorType::Str(b)) => a.iter().zip(b.iter()).all(|(a, b)| a == b),
            (TensorType::Bool(a), TensorType::Bool(b)) => {
                a.iter().zip(b.iter()).all(|(a, b)| a == b)
            }
            _ => false,
        }
    }
}

fn allclose<A: ArrayElement + num::Signed>(
    lhs: &ArrayD<A>,
    rhs: &ArrayD<A>,
    rtol: Option<f32>,
    atol: Option<f32>,
    equal_nan: Option<bool>,
) -> bool
where
    f32: F32IntoType<A>,
{
    let rtol: A = F32IntoType::as_(rtol.unwrap_or(1e-5));
    let atol: A = F32IntoType::as_(atol.unwrap_or(1e-8));
    let equal_nan = equal_nan.unwrap_or(false);
    lhs.iter().zip(rhs.iter()).all(|(&l, &r)| {
        if l.is_nan() && r.is_nan() {
            equal_nan
        } else {
            (l - r).abs() <= atol + rtol * r.abs()
        }
    })
}

fn allclose_unsigned<A: ArrayElement>(
    lhs: &ArrayD<A>,
    rhs: &ArrayD<A>,
    rtol: Option<f32>,
    atol: Option<f32>,
    equal_nan: Option<bool>,
) -> bool
where
    f32: F32IntoType<A>,
{
    let rtol: A = F32IntoType::as_(rtol.unwrap_or(1e-5));
    let atol: A = F32IntoType::as_(atol.unwrap_or(1e-8));
    let equal_nan = equal_nan.unwrap_or(false);
    lhs.iter().zip(rhs.iter()).all(|(&l, &r)| {
        if l.is_nan() && r.is_nan() {
            equal_nan
        } else {
            (l - r) <= atol + rtol * r
        }
    })
}
