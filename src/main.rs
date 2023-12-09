mod common;
mod executor;
mod onnxparser;
mod operators;
mod protograph;
mod utils;

use anyhow::anyhow;
use common::{BoxResult, VERBOSE};
pub use onnxparser::onnx;
use std::{path::Path, time::Instant};
pub use utils::{initialize_nodes, make_initializers, read_model, read_tensor};

use clap::Parser;

use utils::{make_external_outputs, make_graph_outputs};

use crate::{
    common::{Args, FileInputs, VerbosityLevel, MAX_OPSET_VERSION},
    executor::{compare_outputs, create_links_and_requirements, handle_output},
    protograph::{build_graph_from_proto, GraphOutputType},
};
use std::sync::mpsc::channel;

fn main() -> BoxResult<()> {
    let args = Args::parse();
    VERBOSE
        .set(VerbosityLevel::new(args.verbose as usize))
        .map_err(|_| anyhow!("Failed to set verbosity"))?;
    print_at_level!(
        VerbosityLevel::Minimal,
        "Running model: {}",
        args.model.display()
    );
    let inputspath = if args.model.is_relative() {
        Path::new("models").join(&args.model).join("inputs.json")
    } else {
        args.model.join("inputs.json")
    };
    let inputs_file = std::fs::File::open(inputspath)?;
    let mut fileinputs: FileInputs = serde_json::from_reader(inputs_file)?;
    fileinputs.extend_paths(&args.model);
    let model = read_model(Path::new(&fileinputs.modelpath))?;
    let outputs_dir = Path::new("outputs").join(&args.model);
    if VERBOSE
        .get()
        .map_or(false, |&v| v >= VerbosityLevel::Results)
    {
        std::fs::create_dir_all(&outputs_dir)?;
    }

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
        return Err(anyhow!(
            "Opset version {} is not supported, max supported version is {}",
            opset_version,
            MAX_OPSET_VERSION
        ));
    }
    let start = Instant::now();
    for graph in model.graph.iter() {
        if args.gengraph {
            build_graph_from_proto(
                graph,
                &fileinputs.modelpath,
                match args.graphtype.as_str() {
                    "json" => GraphOutputType::Json,
                    "dot" => GraphOutputType::Dot,
                    _ => return Err(anyhow!("Invalid graph type")),
                },
            )?;
        }
        let initializers = make_initializers(graph)?;
        let node_inputs = initialize_nodes(graph, &fileinputs, initializers)?;
        let expected_outputs = make_external_outputs(graph, &fileinputs)?;
        let mut graph_outputs = make_graph_outputs(graph)?;
        let mut dependency_graph = create_links_and_requirements(graph, node_inputs)?;
        for vi in graph.value_info.iter() {
            if let Some(onnx::type_proto::Value::TensorType(_)) = vi.type_.value {
                // If the type is Tensor, then we are fine because that's implemented
            } else {
                unimplemented!("ValueInfoProto type {:?}", vi.type_)
            }
        }

        print_at_level!(
            VerbosityLevel::Informational,
            "Number of not implemented operators: {}",
            dependency_graph.not_implemented.len()
        );
        for name in dependency_graph.not_implemented.iter() {
            eprintln!("Model uses operator {} which is not implemented yet", name);
        }
        if !dependency_graph.not_implemented.is_empty() && args.failfast {
            return Err(anyhow!("Not implemented operators found"));
        }
        let (tx, rx) = channel();
        loop {
            let mut nodes_ready = vec![];
            for (node, inputs) in dependency_graph.node_input_requirements.iter() {
                if inputs.is_empty() {
                    nodes_ready.push(node.clone());
                }
            }
            dependency_graph
                .node_input_requirements
                .retain(|_, v| !v.is_empty());
            let node_inputs_ref = &dependency_graph.node_inputs;
            rayon::scope(|s| {
                for node in nodes_ready {
                    let tx = tx.clone();
                    s.spawn(move |_| {
                        let result = node.execute(node_inputs_ref, opset_version);
                        tx.send((node, result)).unwrap();
                    })
                }
            });

            let node_inputs = &mut dependency_graph.node_inputs;
            loop {
                match rx.try_recv() {
                    Ok((node, result)) => {
                        let result = result?;
                        let outputs = handle_output(
                            result,
                            &node,
                            &outputs_dir,
                            node_inputs,
                            &mut graph_outputs,
                        )?;
                        for output in outputs {
                            if let Some(n) = dependency_graph.input_link_map.remove(output) {
                                for node in n {
                                    dependency_graph
                                        .node_input_requirements
                                        .entry(node)
                                        .and_modify(|v| {
                                            v.retain(|x| *x != output);
                                        });
                                }
                            }
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        return Err(anyhow!("Channel disconnected"));
                    }
                }
            }
            if dependency_graph.node_input_requirements.is_empty() {
                break;
            }
        }
        compare_outputs(expected_outputs, &mut graph_outputs)?;
    }
    let duration = start.elapsed();
    print_at_level!(
        VerbosityLevel::Minimal,
        "Time elapsed in execution is: {:?}",
        duration
    );
    Ok(())
}
