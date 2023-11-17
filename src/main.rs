mod onnxparser;
mod operators;
mod utils;

use lazy_static::lazy_static;
pub use onnxparser::onnx;
use std::collections::HashMap;
use utils::BoxResult;
pub use utils::{make_external_inputs, make_initializers, read_model, read_tensor, OperationFn};

use operators::add::add;
use operators::clip::clip;
use operators::concat::concat;
use operators::constant::constant;
use operators::conv::conv;
use operators::gather::gather;
use operators::gemm::gemm;
use operators::globalaveragepool::global_average_pool;
use operators::lrn::lrn;
use operators::maxpool::maxpool;
use operators::relu::relu;
use operators::reshape::reshape;
use operators::shape::shape;
use operators::unsqueeze::unsqueeze;
use std::path::{Path, PathBuf};

use clap::Parser;

use serde::{Deserialize, Serialize};

use crate::utils::{make_external_outputs, make_graph_outputs, ArrayType, OperationResult};

lazy_static! {
    static ref OPERATION_MAP: HashMap<&'static str, OperationFn> = {
        let mut m = HashMap::new();
        m.insert("Conv", conv as OperationFn);
        m.insert("Clip", clip as OperationFn);
        m.insert("Add", add as OperationFn);
        m.insert("GlobalAveragePool", global_average_pool as OperationFn);
        m.insert("Shape", shape as OperationFn);
        m.insert("Constant", constant as OperationFn);
        m.insert("Gather", gather as OperationFn);
        m.insert("Unsqueeze", unsqueeze as OperationFn);
        m.insert("Concat", concat as OperationFn);
        m.insert("Reshape", reshape as OperationFn);
        m.insert("Gemm", gemm as OperationFn);
        m.insert("Relu", relu as OperationFn);
        m.insert("LRN", lrn as OperationFn);
        m.insert("MaxPool", maxpool as OperationFn);
        m
    };
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    pub model: PathBuf,
}

#[derive(Serialize, Deserialize)]
pub struct FileInputs {
    pub inputs: Vec<PathBuf>,
    pub outputs: Vec<PathBuf>,
    pub modelpath: PathBuf,
}

impl FileInputs {
    fn extend_paths(&mut self, modelname: &Path) {
        self.inputs = self
            .inputs
            .iter()
            .map(|s| Path::new("models").join(modelname).join(s))
            .collect();
        self.outputs = self
            .outputs
            .iter()
            .map(|s| Path::new("models").join(modelname).join(s))
            .collect();
        self.modelpath = Path::new("models")
            .join(modelname)
            .join(self.modelpath.clone())
    }
}

const MAX_OPSET_VERSION: i64 = 20;

fn main() -> BoxResult<()> {
    let args = Args::parse();
    println!("Running model: {}", args.model.display());
    let inputspath = Path::new("models").join(&args.model).join("inputs.json");
    let inputs_file = std::fs::File::open(&inputspath)?;
    let mut fileinputs: FileInputs = serde_json::from_reader(inputs_file)?;
    fileinputs.extend_paths(&args.model);
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
            if let Some(name) = node.op_type.as_deref() {
                if OPERATION_MAP.get(name).is_none() {
                    eprintln!("Model uses operator {} which is not implemented yet", name);
                }
            }
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
            if let Some(op_name) = node.op_type.as_deref() {
                if let Some(func) = OPERATION_MAP.get(op_name) {
                    println!(
                        "Running {} operator between {:?} to get {:?}",
                        op_name, input_names, output_names
                    );
                    let result = func(&inputs, node, opset_version, outputs.len())?;
                    match result {
                        OperationResult::Double((a, b)) => {
                            assert_eq!(outputs.len(), 2);
                            let output_name = outputs[0];
                            let output_name2 = outputs[1];
                            node_inputs.insert(output_name.to_string(), a);
                            node_inputs.insert(output_name2.to_string(), b);
                        }
                        OperationResult::Single(res) => {
                            assert_eq!(outputs.len(), 1);
                            let output_name = outputs[0];
                            node_inputs.insert(output_name.to_string(), res);
                        }
                        OperationResult::OptionalDouble((a, Some(b))) => {
                            assert_eq!(outputs.len(), 2);
                            let output_name = outputs[0];
                            let output_name2 = outputs[1];
                            node_inputs.insert(output_name.to_string(), a);
                            node_inputs.insert(output_name2.to_string(), b);
                        }
                        OperationResult::OptionalDouble((a, None)) => {
                            assert_eq!(outputs.len(), 1);
                            let output_name = outputs[0];
                            node_inputs.insert(output_name.to_string(), a);
                        }
                        OperationResult::Multiple(res) => {
                            assert_eq!(outputs.len(), res.len());
                            for (i, output_name) in outputs.iter().enumerate() {
                                node_inputs.insert(output_name.to_string(), res[i].clone());
                            }
                        }
                    }
                } else {
                    todo!("Op type {:?}", op_name)
                }
            } else {
                todo!("Node {:?} doesn't have op type", node)
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
                        println!("Output {} has shape {:?} as expected", name, value.shape());
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
                        _ => todo!(
                            "Compare output {:?} with {:?}",
                            value.value_type(),
                            data.value_type()
                        ),
                    }
                }
            }
        }
    }
    Ok(())
}
