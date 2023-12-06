mod common;
mod onnxparser;
mod operators;
mod protograph;
mod utils;

use anyhow::anyhow;
use common::{ArrayType, BoxResult, OperationFn, OperationResult, VERBOSE};
use lazy_static::lazy_static;
pub use onnxparser::onnx;
use std::collections::{HashMap, HashSet};
pub use utils::{initialize_nodes, make_initializers, read_model, read_tensor};

use operators::add::add;
use operators::averagepool::averagepool;
use operators::batchnormalization::batchnormalization;
use operators::cast::cast;
use operators::clip::clip;
use operators::concat::concat;
use operators::constant::constant;
use operators::constantofshape::constantofshape;
use operators::conv::conv;
use operators::div::div;
use operators::dropout::dropout;
use operators::exp::exp;
use operators::flatten::flatten;
use operators::gather::gather;
use operators::gemm::gemm;
use operators::globalaveragepool::global_average_pool;
use operators::lrn::lrn;
use operators::matmul::matmul;
use operators::maxpool::maxpool;
use operators::mul::mul;
use operators::nonzero::nonzero;
use operators::pow::pow;
use operators::reducemean::reducemean;
use operators::relu::relu;
use operators::reshape::reshape;
use operators::shape::shape;
use operators::slice::slice;
use operators::softmax::softmax;
use operators::split::split;
use operators::sqrt::sqrt;
use operators::squeeze::squeeze;
use operators::sub::sub;
use operators::sum::sum;
use operators::tanh::tanh;
use operators::transpose::transpose;
use operators::unsqueeze::unsqueeze;
use protograph::build_graph_from_proto;
use std::path::{Path, PathBuf};

use clap::Parser;

use serde::{Deserialize, Serialize};

use utils::{make_external_outputs, make_graph_outputs, OutputInfo};

use crate::utils::operator_not_implemented;
use std::sync::mpsc::channel;

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
        m.insert("Softmax", softmax as OperationFn);
        m.insert("Dropout", dropout as OperationFn);
        m.insert("Sub", sub as OperationFn);
        m.insert("Div", div as OperationFn);
        m.insert("ConstantOfShape", constantofshape as OperationFn);
        m.insert("NonZero", nonzero as OperationFn);
        m.insert("AveragePool", averagepool as OperationFn);
        m.insert("Transpose", transpose as OperationFn);
        m.insert("Sqrt", sqrt as OperationFn);
        m.insert("Mul", mul as OperationFn);
        m.insert("Pow", pow as OperationFn);
        m.insert("Squeeze", squeeze as OperationFn);
        m.insert("Exp", exp as OperationFn);
        m.insert("Tanh", tanh as OperationFn);
        m.insert("Split", split as OperationFn);
        m.insert("MatMul", matmul as OperationFn);
        m.insert("ReduceMean", reducemean as OperationFn);
        m.insert("Slice", slice as OperationFn);
        m.insert("BatchNormalization", batchnormalization as OperationFn);
        m.insert("Cast", cast as OperationFn);
        m.insert("Sum", sum as OperationFn);
        m.insert("Flatten", flatten as OperationFn);
        m
    };
}

#[derive(Parser, Debug)]
/// Parse and execute inference on pre-trained ONNX models
struct Args {
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
    fn extend_paths(&mut self, modelname: &Path) {
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

const MAX_OPSET_VERSION: i64 = 20;

#[derive(Debug, Clone, PartialEq)]
struct ONNXNode<'a> {
    id: usize,
    op_func: OperationFn,
    node_ref: &'a onnx::NodeProto,
}

impl std::hash::Hash for ONNXNode<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl std::cmp::Eq for ONNXNode<'_> {}

fn execute_node(
    onnxnode: &ONNXNode,
    node_inputs: &HashMap<String, ArrayType>,
    opset_version: i64,
) -> BoxResult<OperationResult> {
    let node = onnxnode.node_ref;
    let mut inputs = vec![];
    let mut outputs = vec![];
    let mut all_nodes_have_init = true;
    for input in node.input.iter() {
        if let Some(k) = node_inputs.get(input) {
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
            Ok(result)
        } else {
            Err(anyhow!("Op type not implemented {:?}", op_name))
        }
    } else {
        Err(anyhow!("Op type not found"))
    }
}

fn handle_output<'a>(
    result: OperationResult,
    node: &'a ONNXNode,
    args: &Args,
    outputs_dir: &Path,
    node_inputs: &mut HashMap<String, ArrayType>,
    graph_outputs: &mut HashMap<String, OutputInfo>,
) -> BoxResult<Vec<&'a str>> {
    let node = node.node_ref;
    let outputs = node
        .output
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<&str>>();
    match result {
        OperationResult::Double((a, b)) => {
            assert_eq!(outputs.len(), 2);
            let output_name = outputs[0];
            let output_name2 = outputs[1];
            println!("\tOutput {} has shape {:?}", output_name, a.shape());
            println!("\tOutput {} has shape {:?}", output_name2, b.shape());
            if args.verbose >= 2 {
                a.to_file(outputs_dir, output_name)?;
                b.to_file(outputs_dir, output_name2)?;
            }
            node_inputs.insert(output_name.to_string(), a);
            node_inputs.insert(output_name2.to_string(), b);
        }
        OperationResult::Single(res) => {
            assert_eq!(outputs.len(), 1);
            let output_name = outputs[0];
            println!("\tOutput {} has shape {:?}", output_name, res.shape());
            if args.verbose >= 2 {
                res.to_file(outputs_dir, output_name)?;
            }
            node_inputs.insert(output_name.to_string(), res);
        }
        OperationResult::OptionalDouble((a, Some(b))) => {
            assert_eq!(outputs.len(), 2);
            let output_name = outputs[0];
            let output_name2 = outputs[1];
            println!("\tOutput {} has shape {:?}", output_name, a.shape());
            println!("\tOutput {} has shape {:?}", output_name2, b.shape());
            if args.verbose >= 2 {
                a.to_file(outputs_dir, output_name)?;
                b.to_file(outputs_dir, output_name2)?;
            }
            node_inputs.insert(output_name.to_string(), a);
            node_inputs.insert(output_name2.to_string(), b);
        }
        OperationResult::OptionalDouble((a, None)) => {
            assert_eq!(outputs.len(), 1);
            let output_name = outputs[0];
            println!("\tOutput {} has shape {:?}", output_name, a.shape());
            if args.verbose >= 2 {
                a.to_file(outputs_dir, output_name)?;
            }
            node_inputs.insert(output_name.to_string(), a);
        }
        OperationResult::Multiple(res) => {
            assert_eq!(outputs.len(), res.len());
            for (output_name, res) in outputs.iter().zip(res.into_iter()) {
                println!("\tOutput {} has shape {:?}", output_name, res.shape());
                if args.verbose >= 2 {
                    res.to_file(outputs_dir, output_name)?;
                }
                node_inputs.insert(output_name.to_string(), res);
            }
        }
    }
    for output_name in outputs.iter() {
        if let Some(gout) = graph_outputs.get_mut(*output_name) {
            if let Some(produced) = node_inputs.get(*output_name) {
                gout.data = Some(produced.clone());
            }
        }
    }
    Ok(outputs)
}

fn main() -> BoxResult<()> {
    let args = Args::parse();
    VERBOSE
        .set(args.verbose as usize)
        .map_err(|_| anyhow!("Failed to set verbosity"))?;
    println!("Running model: {}", args.model.display());
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
    if args.verbose >= 2 {
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
        panic!(
            "Opset version {} is not supported, max supported version is {}",
            opset_version, MAX_OPSET_VERSION
        );
    }
    for graph in model.graph.iter() {
        if args.gengraph {
            build_graph_from_proto(graph, &fileinputs.modelpath)?;
        }
        let mut node_input_requirements: HashMap<ONNXNode, Vec<&str>> = HashMap::new();
        let mut input_link_map: HashMap<&str, Vec<ONNXNode>> = HashMap::new();
        let initializers = make_initializers(graph);
        let mut node_inputs = initialize_nodes(graph, &fileinputs, initializers)?;
        let expected_outputs = make_external_outputs(graph, &fileinputs)?;
        let mut graph_outputs = make_graph_outputs(graph)?;
        let mut not_implemented = HashSet::new();
        for (counter, node) in graph.node.iter().enumerate() {
            let input_names = node
                .input
                .iter()
                .filter_map(|s| {
                    if node_inputs.contains_key(s.as_str()) {
                        None
                    } else {
                        Some(s.as_str())
                    }
                })
                .collect::<Vec<&str>>();
            if let Some(name) = node.op_type.as_deref() {
                for input_name in input_names.iter() {
                    input_link_map
                        .entry(input_name)
                        .or_default()
                        .push(ONNXNode {
                            id: counter,
                            op_func: *OPERATION_MAP
                                .get(name)
                                .unwrap_or(&(operator_not_implemented as OperationFn)),
                            node_ref: node,
                        });
                }
                if let Some(op_func) = OPERATION_MAP.get(name) {
                    node_input_requirements.insert(
                        ONNXNode {
                            id: counter,
                            op_func: *op_func,
                            node_ref: node,
                        },
                        input_names,
                    );
                } else {
                    node_input_requirements.insert(
                        ONNXNode {
                            id: counter,
                            op_func: operator_not_implemented as OperationFn,
                            node_ref: node,
                        },
                        input_names,
                    );
                    not_implemented.insert(name);
                }
            }
        }

        eprintln!(
            "Number of not implemented operators: {}",
            not_implemented.len()
        );
        for name in not_implemented.iter() {
            eprintln!("Model uses operator {} which is not implemented yet", name);
        }
        if !not_implemented.is_empty() && args.failfast {
            return Err(anyhow!("Not implemented operators found"));
        }
        let (tx, rx) = channel();
        loop {
            let mut nodes_ready = vec![];
            for (node, inputs) in node_input_requirements.iter() {
                if inputs.is_empty() {
                    println!("Node {} is ready", node.id);
                    nodes_ready.push(node.clone());
                }
            }
            node_input_requirements.retain(|_, v| !v.is_empty());
            let node_inputs_ref = &node_inputs;
            rayon::scope(|s| {
                for node in nodes_ready {
                    let tx = tx.clone();
                    s.spawn(move |_| {
                        let result = execute_node(&node, node_inputs_ref, opset_version);
                        tx.send((node, result)).unwrap();
                    })
                }
            });

            let node_inputs = &mut node_inputs;
            let (node, result) = rx.recv()?;
            let result = result?;
            let outputs = handle_output(
                result,
                &node,
                &args,
                &outputs_dir,
                node_inputs,
                &mut graph_outputs,
            )?;
            for output in outputs {
                if let Some(n) = input_link_map.remove(output) {
                    for node in n {
                        node_input_requirements.entry(node).and_modify(|v| {
                            v.retain(|x| *x != output);
                        });
                    }
                }
            }
            if node_input_requirements.is_empty() {
                break;
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
                            let mut count = 0;
                            let mut diff = vec![];
                            for (i, (v, d)) in v.iter().zip(d.iter()).enumerate() {
                                if (v - d).abs() > 0.0001 {
                                    count += 1;
                                }
                                diff.push((i, v, d, (v - d).abs()));
                            }
                            let max = diff
                                .iter()
                                .max_by(|(_, _, _, d1), (_, _, _, d2)| {
                                    d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Less)
                                })
                                .expect("Failed to get max difference");
                            println!("Output {} has {} values with absolute difference of more than .0001", name, count);
                            println!("\tMax difference: {:?}", max);
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
