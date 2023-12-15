use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use crate::{
    common::{
        Args, BoxResult, FileInputs, OperationFn, OperatorResult, TensorType, VerbosityLevel,
        MAX_OPSET_VERSION, VERBOSE,
    },
    onnx,
    operators::OPERATION_MAP,
    print_at_level,
    protograph::{build_graph_from_proto, GraphOutputType},
    read_model,
    utils::{initialize_nodes, make_initializers},
    utils::{make_external_outputs, make_graph_outputs, operator_not_implemented, OutputInfo},
};

use anyhow::anyhow;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
#[derive(Debug, Clone, PartialEq)]
/// Represent an ONNX execution node, with an ID, the function to execute it and a reference to the internal ONNX node
struct ONNXNode<'a> {
    id: usize,
    op_func: OperationFn,
    node_ref: &'a onnx::NodeProto,
}

impl<'a> ONNXNode<'a> {
    /// Create a new ONNXNode
    fn new(id: usize, op_func: OperationFn, node_ref: &'a onnx::NodeProto) -> Self {
        Self {
            id,
            op_func,
            node_ref,
        }
    }

    /// Executes the ONNX node given the inputs map and the opset version
    fn execute(
        &self,
        node_inputs: &HashMap<String, TensorType>,
        opset_version: i64,
    ) -> BoxResult<OperatorResult> {
        let mut inputs = vec![];
        let mut outputs = vec![];
        let mut all_nodes_have_init = true;
        for input in self.node_ref.input.iter() {
            if let Some(k) = node_inputs.get(input) {
                inputs.push(k);
            } else {
                all_nodes_have_init = false;
            }
        }
        for output in self.node_ref.output.iter() {
            outputs.push(output);
        }
        if !all_nodes_have_init {
            return Err(anyhow!("Some nodes in this operation have not been initialized yet, this means the operations aren't in order, fix the code to account for this"));
        }
        let input_names = self
            .node_ref
            .input
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        let output_names = self
            .node_ref
            .output
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        print_at_level!(
            VerbosityLevel::Informational,
            "Running {} operator (id: {}, thread: {}) between {:?} to get {:?}",
            self.node_ref.op_type(),
            self.id,
            rayon::current_thread_index()
                .map_or_else(|| "(no thread)".to_string(), |i| i.to_string()),
            input_names,
            output_names
        );
        (self.op_func)(&inputs, self.node_ref, opset_version, output_names.len())
    }
}

impl std::hash::Hash for ONNXNode<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl std::cmp::Eq for ONNXNode<'_> {}

/// Handle the output of an operator, saving it to disk if verbose is VerbosityLevel::Results or higher, and saving it to graph_outputs
fn handle_output<'a>(
    result: OperatorResult,
    node: &'a ONNXNode,
    outputs_dir: &Path,
    node_inputs: &mut HashMap<String, TensorType>,
    graph_outputs: &mut HashMap<String, OutputInfo>,
) -> BoxResult<Vec<&'a str>> {
    let node = node.node_ref;
    let outputs = node
        .output
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<&str>>();
    let result = result.result; // we love rust
    assert_eq!(outputs.len(), result.len());
    for (output_name, res) in outputs.iter().zip(result.into_iter()) {
        print_at_level!(
            VerbosityLevel::Informational,
            "\tOutput {} has shape {:?}",
            output_name,
            res.shape()
        );
        if VERBOSE
            .get()
            .map_or(false, |&v| v >= VerbosityLevel::Results)
        {
            res.to_file(outputs_dir, output_name)?;
        }
        node_inputs.insert(output_name.to_string(), res);
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

/// A dependency graph of ONNX nodes
struct DependencyGraph<'a> {
    /// A map of ONNX nodes to their input requirements
    pub node_input_requirements: HashMap<ONNXNode<'a>, Vec<&'a str>>,
    /// A map of input names to the nodes that require them
    pub input_link_map: HashMap<&'a str, Vec<ONNXNode<'a>>>,
    /// A set of not implemented operators
    pub not_implemented: HashSet<&'a str>,
    /// A map of input names to their corresponding tensors
    pub node_inputs: HashMap<String, TensorType>,
}

/// Create a dependency graph from an ONNX graph and a map of input names to their corresponding tensors
fn create_links_and_requirements(
    graph: &onnx::GraphProto,
    node_inputs: HashMap<String, TensorType>,
) -> BoxResult<DependencyGraph> {
    let mut node_input_requirements: HashMap<ONNXNode, Vec<&str>> = HashMap::new();
    let mut input_link_map: HashMap<&str, Vec<ONNXNode>> = HashMap::new();
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
                    .push(ONNXNode::new(
                        counter,
                        *OPERATION_MAP
                            .get(name)
                            .unwrap_or(&(operator_not_implemented as OperationFn)),
                        node,
                    ));
            }
            if let Some(op_func) = OPERATION_MAP.get(name) {
                node_input_requirements.insert(ONNXNode::new(counter, *op_func, node), input_names);
            } else {
                node_input_requirements.insert(
                    ONNXNode::new(counter, operator_not_implemented as OperationFn, node),
                    input_names,
                );
                not_implemented.insert(name);
            }
        }
    }
    Ok(DependencyGraph {
        node_input_requirements,
        input_link_map,
        not_implemented,
        node_inputs,
    })
}

/// Compare the expected outputs to the actual outputs
pub fn compare_outputs(
    expected_outputs: HashMap<String, TensorType>,
    graph_outputs: &mut HashMap<String, OutputInfo>,
) -> BoxResult<()> {
    for (name, value) in expected_outputs.iter() {
        if let Some(gout) = graph_outputs.get_mut(name) {
            if let Some(data) = &gout.data {
                if value.shape() != data.shape() {
                    return Err(anyhow!(
                        "Expected output {} to have shape {:?} but got {:?}",
                        name,
                        value.shape(),
                        data.shape()
                    ));
                } else {
                    print_at_level!(
                        VerbosityLevel::Minimal,
                        "Output {} has shape {:?} as expected",
                        name,
                        value.shape()
                    );
                }
                if value.value_type() != data.value_type() {
                    return Err(anyhow!(
                        "Expected output {} to have type {:?} but got {:?}",
                        name,
                        value.value_type(),
                        data.value_type()
                    ));
                } else {
                    print_at_level!(
                        VerbosityLevel::Minimal,
                        "Output {} has type {:?} as expected",
                        name,
                        value.value_type()
                    );
                }
                match (value, data) {
                    (TensorType::F32(v), TensorType::F32(d)) => {
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
                        print_at_level!(
                            VerbosityLevel::Minimal,
                            "Output {} has {} values with absolute difference of more than .0001\n\tMax difference: {:?}",
                            name,
                            count,
                            max
                        );
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
    Ok(())
}

pub fn execute_model(args: &Args) -> BoxResult<()> {
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
            let mut results = vec![];
            nodes_ready
                .into_par_iter()
                .map(|n| {
                    let r = n.execute(node_inputs_ref, opset_version);
                    (r, n)
                })
                .collect_into_vec(&mut results);

            let node_inputs = &mut dependency_graph.node_inputs;
            results
                .into_iter()
                .try_for_each(|(result, node)| -> Result<(), anyhow::Error> {
                    let outputs = handle_output(
                        result?,
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
                    Ok(())
                })?;
            if dependency_graph.node_input_requirements.is_empty() {
                break;
            }
        }
        compare_outputs(expected_outputs, &mut graph_outputs)?;
    }
    Ok(())
}
