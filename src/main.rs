mod onnxparser;
mod operators;
mod utils;

pub use onnxparser::onnx;
pub use utils::{make_external_inputs, make_initializers, make_inputs, read_model};

use operators::add::add;
use operators::clip::clip;
use operators::constant::constant;
use operators::conv::conv;
use operators::globalaveragepool::global_average_pool;
use operators::shape::shape;
use operators::gather::gather;
use std::path::Path;

use clap::Parser;

use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "inputs.json")]
    pub inputs_file: String,
}

#[derive(Serialize, Deserialize)]
pub struct FileInputs {
    pub inputs: Vec<FileInput>,
    pub modelpath: String,
}

#[derive(Serialize, Deserialize)]
pub struct FileInput {
    pub name: String,
    pub datafile: String,
    pub attributes: serde_json::Map<String, serde_json::Value>,
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
        let external_inputs = make_external_inputs(graph, &fileinputs);
        let mut node_inputs = make_inputs(graph, &external_inputs, &fileinputs);
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
                Some(n) => {
                    todo!("Op type {:?}", n)
                }
                None => {
                    panic!("  Node has no op type");
                }
            }
        }
    }
    Ok(())
}
