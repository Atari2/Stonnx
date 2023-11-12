mod operators;
mod onnxparser;
mod utils;

pub use onnxparser::onnx;
pub use utils::utils::{read_model_binary, make_external_inputs, make_initializers, make_inputs};

use operators::conv::conv;
use operators::clip::clip;
use std::path::Path;

use clap::Parser;

use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "inputs.json")]
    pub inputs_file: String
}

#[derive(Serialize, Deserialize)]
pub struct FileInputs {
    pub inputs: Vec<FileInput>
}

#[derive(Serialize, Deserialize)]
pub struct FileInput {
    pub name: String,
    pub datafile: String,
    pub attributes: serde_json::Map<String, serde_json::Value>
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("Inputs file: {}", args.inputs_file);
    let inputs_file = std::fs::File::open(args.inputs_file)?;
    let fileinputs: FileInputs = serde_json::from_reader(inputs_file)?;
    let model = read_model_binary(Path::new("examples/mobilenetv2-10.onnx"))?;
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
                    inputs.push(i.view());
                } else if let Some(k) = node_inputs.get(input) {
                    inputs.push(k.view());
                } else {
                    all_nodes_have_init = false;
                }
            }
            for output in node.output.iter() {
                outputs.push(output);
            }
            if !all_nodes_have_init {
                continue;
            }
            match node.op_type.as_deref() {
                Some("Conv") => {
                    let input_names = node.input.iter().map(|s| s.as_str()).collect::<Vec<&str>>();
                    let output_names = node.output.iter().map(|s| s.as_str()).collect::<Vec<&str>>();
                    println!("Running conv operator between {:?} to get {:?}", input_names, output_names);
                    let conv_result = conv(&inputs, node)?;
                    assert_eq!(outputs.len(), 1);
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), conv_result);
                }
                Some("Clip") => {
                    let input_names = node.input.iter().map(|s| s.as_str()).collect::<Vec<&str>>();
                    let output_names = node.output.iter().map(|s| s.as_str()).collect::<Vec<&str>>();
                    println!("Running clip operator between {:?} to get {:?}", input_names, output_names);
                    let clip_result = clip(&inputs)?;
                    assert_eq!(outputs.len(), 1);
                    let output_name = outputs[0];
                    node_inputs.insert(output_name.to_string(), clip_result);
                }
                Some(n) => {
                    todo!("Op type {:?} not implemented", n)
                }
                None => {
                    panic!("  Node has no op type");
                }
            }
        }
    }
    // for entry in WalkDir::new("examples") {
    //     let entry = entry?;
    //     if entry.path().extension().map_or(false, |e| e == "onnx") {
    //         match read_model_binary(entry.path()) {
    //             Ok(_) => println!("{}: OK", entry.path().display()),
    //             Err(e) => {
    //                 println!("Error during binary parsing of {}: {}, trying with text", entry.path().display(), e);
    //                 match read_model_text(entry.path()) {
    //                     Ok(_) => println!("{}: OK", entry.path().display()),
    //                     Err(e) => println!("Error during text parsing of {}: {}", entry.path().display(), e),
    //                 }
    //             },
    //         }
    //     }
    // }

    Ok(())
}
