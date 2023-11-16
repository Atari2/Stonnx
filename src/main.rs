mod onnxparser;
mod operators;
mod utils;

pub use onnxparser::onnx;
pub use utils::{make_external_inputs, make_initializers, read_model, read_tensor};

use operators::add::add;
use operators::clip::clip;
use operators::constant::constant;
use operators::conv::conv;
use operators::gather::gather;
use operators::globalaveragepool::global_average_pool;
use operators::shape::shape;
use operators::unsqueeze::unsqueeze;
use operators::concat::concat;
use operators::reshape::reshape;
use operators::gemm::gemm;
use operators::relu::relu;
use operators::lrn::lrn;
use std::path::Path;

use clap::Parser;

use serde::{Deserialize, Serialize};

use crate::utils::{make_external_outputs, make_graph_outputs, ArrayType};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "inputs.json")]
    pub inputs_file: String,
}

#[derive(Serialize, Deserialize)]
pub struct FileInputs {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub modelpath: String
}

const MAX_OPSET_VERSION: i64 = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("Inputs file: {}", args.inputs_file);
    let inputs_file = std::fs::File::open(args.inputs_file)?;
    let fileinputs: FileInputs = serde_json::from_reader(inputs_file)?;
    let model = read_model(Path::new(&fileinputs.modelpath))?;
    let opset_version = if let Some(v) = model.opset_import.get(0) {
        if let Some(v) = v.version {
            v
        } else {
            MAX_OPSET_VERSION
        }
    } else {
        MAX_OPSET_VERSION
    };
    if opset_version > MAX_OPSET_VERSION {
        panic!(
            "Opset version {} is not supported, max supported version is {}",
            opset_version, MAX_OPSET_VERSION
        );
    }
    for graph in model.graph.iter() {
        let initializers = make_initializers(graph);
        let mut node_inputs = make_external_inputs(graph, &fileinputs)?;
        let expected_outputs = make_external_outputs(graph, &fileinputs)?;
        let mut graph_outputs = make_graph_outputs(graph)?;
        for node in graph.node.iter() {
            println!("Node {:?}", node.op_type.as_deref());
        }
        for node in graph.node.iter() {
            let mut inputs = vec![];
            let mut outputs = vec![];
            let mut all_nodes_have_init = true;
            for input in node.input.iter() {
                if let Some(i) = initializers.get(input) {
                    inputs.push(i);
                } else if let Some(k) = node_inputs.get(input) {
                    inputs.push(k);
                } else {
                    all_nodes_have_init = false;
                }
            }
            for output in node.output.iter() {
                outputs.push(output);
            }
            if !all_nodes_have_init {
                panic!("Some nodes in this operation have not been initialized yet, this means the operations aren't in order, fix the code to account for this");
            }
            let input_names = node.input.iter().map(|s| s.as_str()).collect::<Vec<&str>>();
            let output_names = node
                .output
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<&str>>();
            match node.op_type.as_deref() {
                Some("Conv") => {
                    println!(
                        "Running conv operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let conv_result = conv(&inputs, node, opset_version)?;
                    assert_eq!(outputs.len(), 1);
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), conv_result);
                }
                Some("Clip") => {
                    println!(
                        "Running clip operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let clip_result = clip(&inputs, node, opset_version)?;
                    assert_eq!(outputs.len(), 1);
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), clip_result);
                }
                Some("Add") => {
                    println!(
                        "Running add operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let add_result = add(&inputs, node, opset_version)?;
                    assert_eq!(outputs.len(), 1);
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), add_result);
                }
                Some("GlobalAveragePool") => {
                    println!(
                        "Running global average pool operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let gap_result = global_average_pool(&inputs, node, opset_version)?;
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), gap_result);
                }
                Some("Shape") => {
                    println!(
                        "Running shape operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let shape_result = shape(&inputs, node, opset_version)?;
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), shape_result);
                }
                Some("Constant") => {
                    println!(
                        "Running constant operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let constant = constant(&inputs, node, opset_version)?;
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), constant);
                }
                Some("Gather") => {
                    println!(
                        "Running gather operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let gather = gather(&inputs, node, opset_version)?;
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), gather);
                }
                Some("Unsqueeze") => {
                    println!(
                        "Running unsqueeze operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let unsqueeze = unsqueeze(&inputs, node, opset_version)?;
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), unsqueeze);
                }
                Some("Concat") => {
                    println!(
                        "Running concat operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let concat = concat(&inputs, node, opset_version)?;
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), concat);
                }
                Some("Reshape") => {
                    println!(
                        "Running reshape operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let reshape = reshape(&inputs, node, opset_version)?;
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), reshape);
                }
                Some("Gemm") => {
                    println!(
                        "Running gemm operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let reshape = gemm(&inputs, node, opset_version)?;
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), reshape);
                }
                Some("Relu") => {
                    println!(
                        "Running relu operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let relu = relu(&inputs, node, opset_version)?;
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), relu);
                }
                Some("lrn") => {
                    println!(
                        "Running lrn operator between {:?} to get {:?}",
                        input_names, output_names
                    );
                    let lrn = lrn(&inputs, node, opset_version)?;
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), lrn);
                }
                Some(n) => {
                    todo!("Op type {:?}", n)
                }
                None => {
                    panic!("  Node has no op type");
                }
            }
            for output_name in outputs.iter() {
                if let Some(gout) = graph_outputs.get_mut(*output_name) {
                    if let Some(produced) = node_inputs.get(*output_name) {
                        gout.data = Some(produced.clone());
                    }
                }
            }
        }
        for (name, value) in expected_outputs.iter() {
            if let Some(gout) = graph_outputs.get_mut(name) {
                if let Some(data) = &gout.data {
                    if value.shape() != data.shape() {
                        panic!(
                            "Expected output {} to have shape {:?} but got {:?}",
                            name,
                            value.shape(),
                            data.shape()
                        );
                    } else {
                        println!(
                            "Output {} has shape {:?} as expected",
                            name,
                            value.shape()
                        );
                    }
                    if value.value_type() != data.value_type() {
                        panic!(
                            "Expected output {} to have type {:?} but got {:?}",
                            name,
                            value.value_type(),
                            data.value_type()
                        );
                    } else {
                        println!(
                            "Output {} has type {:?} as expected",
                            name,
                            value.value_type()
                        );
                    }
                    match (value, data) {
                        (ArrayType::F32(v), ArrayType::F32(d)) => {
                            let mut diff = vec![];
                            for (i, (v, d)) in v.iter().zip(d.iter()).enumerate() {
                                if (v - d).abs() > 0.0001 {
                                    println!(
                                        "Compare output {:?} with {:?} failed at index {}",
                                        v, d, i
                                    );
                                }
                                diff.push((i, v, d, (v - d).abs()));
                            }
                        }
                        _ => todo!("Compare output {:?} with {:?}", value.value_type(), data.value_type()),
                    }
                }
            }
        }
    }
    Ok(())
}
