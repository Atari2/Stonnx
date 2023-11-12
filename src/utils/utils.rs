use std::io::Read;
use std::{error::Error, path::Path, collections::HashMap, io};

use ndarray::{ArrayViewD, IxDyn};
use protobuf::Enum;

use crate::onnxparser::onnx;
use crate::onnxparser::onnx::tensor_proto;
use crate::onnxparser::onnx::tensor_shape_proto::Dimension;

static UNKNOWN: &str = "<unknown>";

pub enum ArrayType<'a> {
    I64(ArrayViewD<'a, i64>),
    F32(ArrayViewD<'a, f32>),
}

impl<'a> ArrayType<'a> {
    pub fn shape(&self) -> &[usize] {
        match self {
            ArrayType::I64(a) => a.shape(),
            ArrayType::F32(a) => a.shape(),
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

pub fn read_model_binary(p: &Path) -> Result<onnx::ModelProto, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(p)?;
    let mut reader = std::io::BufReader::new(file);
    let model: onnx::ModelProto = protobuf::Message::parse_from_reader(&mut reader)?;
    Ok(model)
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

pub fn get_dim(dim: &Dimension) -> i64 {
    if dim.has_dim_value() {
        dim.dim_value()
    } else if dim.has_dim_param() {
        let mut input_line = String::new();
        println!("Enter {} for input {}", dim.dim_param(), dim.denotation());
        io::stdin().read_line(&mut input_line).unwrap_or_default();
        let x: i64 = input_line.trim().parse().unwrap_or_default();
        x
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

pub fn make_external_inputs(graph: &onnx::GraphProto) -> HashMap<String, Vec<u8>> {
    let mut map = HashMap::new();
    for input in graph.input.iter() {
        let input_name = input.name.as_ref().map_or(UNKNOWN, |v| v.as_str());
        map.insert(input_name.to_string(), get_bytedata(input_name));
    }
    map
}

pub fn make_inputs<'a>(
    graph: &'a onnx::GraphProto,
    external_data: &'a HashMap<String, Vec<u8>>,
) -> HashMap<String, ArrayType<'a>> {
    let mut inputs: HashMap<String, ArrayType<'a>> = HashMap::new();
    for input in graph.input.iter() {
        let input_name = input.name.as_ref().map_or(UNKNOWN, |v| v.as_str());
        match &input.type_.value {
            Some(onnx::type_proto::Value::TensorType(t)) => match (t.elem_type, &t.shape) {
                (Some(et), protobuf::MessageField(Some(s))) => {
                    match make_tensor(
                        &s.dim.iter().map(get_dim).collect::<Vec<i64>>(),
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

pub fn read_model_text(p: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(p)?;
    let mut reader = std::io::BufReader::new(file);
    let mut buf = String::new();
    reader.read_to_string(&mut buf)?;
    let _model: onnx::ModelProto = protobuf::text_format::parse_from_str(&buf)?;
    Ok(())
}