use std::io::Read;
use std::{error::Error, path::Path, collections::HashMap, io};

use ndarray::{ArrayViewD, IxDyn, ArrayD};
use protobuf::Enum;

use crate::{FileInputs, FileInput};
use crate::onnxparser::onnx;
use crate::onnxparser::onnx::tensor_proto;
use crate::onnxparser::onnx::tensor_shape_proto::Dimension;

static UNKNOWN: &str = "<unknown>";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueType {
    I64,
    F32
}


#[derive(Debug)]
pub enum ArrayType<'a> {
    OwnI64(ArrayD<i64>),
    OwnF32(ArrayD<f32>),
    I64(ArrayViewD<'a, i64>),
    F32(ArrayViewD<'a, f32>),
}

impl std::fmt::Display for ArrayType<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrayType::OwnI64(_) => write!(f, "OwnI64"),
            ArrayType::OwnF32(_) => write!(f, "OwnF32"),
            ArrayType::I64(_) => write!(f, "I64"),
            ArrayType::F32(_) => write!(f, "F32"),
        }
    }
}

impl<'a> ArrayType<'a> {
    pub fn shape(&self) -> &[usize] {
        match self {
            ArrayType::OwnI64(a) => a.shape(),
            ArrayType::OwnF32(a) => a.shape(),
            ArrayType::I64(a) => a.shape(),
            ArrayType::F32(a) => a.shape(),
        }
    }
    pub fn to_owned(&self) -> ArrayType<'a> {
        match self {
            ArrayType::OwnI64(a) => ArrayType::OwnI64(a.clone()),
            ArrayType::OwnF32(a) => ArrayType::OwnF32(a.clone()),
            ArrayType::I64(a) => ArrayType::OwnI64(a.to_owned()),
            ArrayType::F32(a) => ArrayType::OwnF32(a.to_owned()),
        }
    }
    pub fn array_view(&'a self) -> ArrayType<'a> {
        match self {
            ArrayType::OwnI64(a) => ArrayType::I64(a.view()),
            ArrayType::OwnF32(a) => ArrayType::F32(a.view()),
            ArrayType::I64(a) => ArrayType::I64(a.view()),
            ArrayType::F32(a) => ArrayType::F32(a.view()),
        }
    }
    pub fn value_type(&self) -> ValueType {
        match self {
            ArrayType::OwnI64(_) => ValueType::I64,
            ArrayType::OwnF32(_) => ValueType::F32,
            ArrayType::I64(_) => ValueType::I64,
            ArrayType::F32(_) => ValueType::F32,
        }
    }
}

pub fn make_tensor<'a>(
    shape: &[i64],
    bytedata: &'a [u8],
    data_type: i32,
) -> Result<ArrayType<'a>, Box<dyn Error>> {
    use tensor_proto::DataType;
    let enum_dt = DataType::from_i32(data_type).unwrap_or_default();
    let shape = shape.iter().map(|v| *v as usize).collect::<Vec<usize>>();
    match enum_dt {
        DataType::UNDEFINED => Err("Undefined data type".into()),
        DataType::INT64 => match bytemuck::try_cast_slice::<u8, i64>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayViewD::<i64>::from_shape(IxDyn(&shape), data)?;
                Ok(ArrayType::I64(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        DataType::FLOAT => match bytemuck::try_cast_slice::<u8, f32>(bytedata) {
            Ok(data) => {
                assert_eq!(data.len(), shape.iter().product::<usize>());
                let a = ArrayViewD::<f32>::from_shape(IxDyn(&shape), data)?;
                Ok(ArrayType::F32(a))
            }
            Err(e) => Err(e.to_string().into()),
        },
        default => {
            todo!("Data type {:?} not implemented", default);
        }
    }
}

pub fn make_initializers<'a>(graph: &'a onnx::GraphProto) -> HashMap<String, ArrayType<'a>> {
    let mut initializers: HashMap<String, ArrayType<'a>> = HashMap::new();
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

pub fn make_external_inputs(graph: &onnx::GraphProto, inputs_file: &FileInputs) -> HashMap<String, Vec<u8>> {
    let mut map = HashMap::new();
    for input in graph.input.iter() {
        let input_name = input.name.as_ref().map_or(UNKNOWN, |v| v.as_str());
        if let Some(input_from_file) = inputs_file.inputs.iter().find(|i| i.name == input_name) {
            map.insert(input_name.to_string(), get_bytedata_from_file(&input_from_file.datafile));
        } else {
            map.insert(input_name.to_string(), get_bytedata(input_name));
        }
    }
    map
}

pub fn make_inputs<'a>(
    graph: &'a onnx::GraphProto,
    external_data: &'a HashMap<String, Vec<u8>>,
    inputs_file: &FileInputs
) -> HashMap<String, ArrayType<'a>> {
    let mut inputs: HashMap<String, ArrayType<'a>> = HashMap::new();
    for input in graph.input.iter() {
        let input_name = input.name.as_ref().map_or(UNKNOWN, |v| v.as_str());
        match &input.type_.value {
            Some(onnx::type_proto::Value::TensorType(t)) => match (t.elem_type, &t.shape) {
                (Some(et), protobuf::MessageField(Some(s))) => {
                    match make_tensor(
                        &s.dim.iter().map(|d| get_dim(d, inputs_file.inputs.iter().find(|i| i.name == input_name))).collect::<Vec<i64>>(),
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
                    println!(
                        "  Input: {} has tensor type but no element type",
                        input_name
                    )
                }
            },
            Some(_) => {
                println!("  Input: {} has type but not tensor type", input_name)
            }
            None => {
                println!("  Input: {} has no type", input_name)
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