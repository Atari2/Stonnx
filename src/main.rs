mod operators;
mod onnxparser;
mod utils;

pub use onnxparser::onnx;
pub use utils::utils::{read_model_binary, make_external_inputs, make_initializers, make_inputs};

use operators::conv::conv;
use std::path::Path;

use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "definitions.json")]
    pub definitions: String
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("Definitions: {}", args.definitions);
    let model = read_model_binary(Path::new("examples/mobilenetv2-10.onnx"))?;
    for graph in model.graph.iter() {
        let initializers = make_initializers(graph);
        let external_inputs = make_external_inputs(graph);
        let node_inputs = make_inputs(graph, &external_inputs);
        for node in graph.node.iter() {
            let mut inputs = vec![];
            let mut outputs = vec![];
            let mut all_nodes_have_init = true;
            for input in node.input.iter() {
                if let Some(i) = initializers.get(input) {
                    println!("  Input: {} is initializer", input);
                    inputs.push(i);
                } else {
                    println!("  Input: {} is not initializer", input);
                    if let Some(k) = node_inputs.get(input) {
                        inputs.push(k);
                    } else {
                        all_nodes_have_init = false;
                    }
                }
            }
            for output in node.output.iter() {
                println!("  Output: {}", output);
                outputs.push(output);
            }
            if !all_nodes_have_init {
                continue;
            }
            match node.op_type.as_deref() {
                Some("Conv") => {
                    conv(&inputs, &mut outputs, node);
                }
                Some(n) => {
                    println!("Op type {:?} not implemented", n)
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
