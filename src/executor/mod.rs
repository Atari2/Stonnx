use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use crate::{
    common::{BoxResult, OperationFn, OperatorResult, TensorType, VerbosityLevel, VERBOSE},
    onnx,
    operators::OPERATION_MAP,
    print_at_level,
    utils::{operator_not_implemented, OutputInfo},
};

use anyhow::anyhow;

#[derive(Debug, Clone, PartialEq)]
/// Represent an ONNX execution node, with an ID, the function to execute it and a reference to the internal ONNX node
pub struct ONNXNode<'a> {
    id: usize,
    op_func: OperationFn,
    node_ref: &'a onnx::NodeProto,
}

impl<'a> ONNXNode<'a> {
    /// Create a new ONNXNode
    pub fn new(id: usize, op_func: OperationFn, node_ref: &'a onnx::NodeProto) -> Self {
        Self {
            id,
            op_func,
            node_ref,
        }
    }

    /// Executes the ONNX node given the inputs map and the opset version
    pub fn execute(
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
pub fn handle_output<'a>(
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
pub struct DependencyGraph<'a> {
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
pub fn create_links_and_requirements(
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
