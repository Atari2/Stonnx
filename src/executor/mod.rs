use std::{collections::HashMap, path::Path};

use crate::{
    common::{ArrayType, BoxResult, OperationFn, OperationResult, VerbosityLevel},
    onnx, print_at_level,
    utils::OutputInfo,
    Args, OPERATION_MAP,
};

use anyhow::anyhow;

#[derive(Debug, Clone, PartialEq)]
pub struct ONNXNode<'a> {
    pub id: usize,
    pub op_func: OperationFn,
    pub node_ref: &'a onnx::NodeProto,
}

impl std::hash::Hash for ONNXNode<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl std::cmp::Eq for ONNXNode<'_> {}

pub fn execute_node(
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
        return Err(anyhow!("Some nodes in this operation have not been initialized yet, this means the operations aren't in order, fix the code to account for this"));
    }
    let input_names = node.input.iter().map(|s| s.as_str()).collect::<Vec<&str>>();
    let output_names = node
        .output
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<&str>>();
    if let Some(op_name) = node.op_type.as_deref() {
        if let Some(func) = OPERATION_MAP.get(op_name) {
            print_at_level!(
                VerbosityLevel::Informational,
                "Running {} operator between {:?} to get {:?}",
                op_name,
                input_names,
                output_names
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

pub fn handle_output<'a>(
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
            print_at_level!(
                VerbosityLevel::Informational,
                "\tOutput {} has shape {:?}",
                output_name,
                a.shape()
            );
            print_at_level!(
                VerbosityLevel::Informational,
                "\tOutput {} has shape {:?}",
                output_name2,
                b.shape()
            );
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
            print_at_level!(
                VerbosityLevel::Informational,
                "\tOutput {} has shape {:?}",
                output_name,
                res.shape()
            );
            if args.verbose >= 2 {
                res.to_file(outputs_dir, output_name)?;
            }
            node_inputs.insert(output_name.to_string(), res);
        }
        OperationResult::OptionalDouble((a, Some(b))) => {
            assert_eq!(outputs.len(), 2);
            let output_name = outputs[0];
            let output_name2 = outputs[1];
            print_at_level!(
                VerbosityLevel::Informational,
                "\tOutput {} has shape {:?}",
                output_name,
                a.shape()
            );
            print_at_level!(
                VerbosityLevel::Informational,
                "\tOutput {} has shape {:?}",
                output_name2,
                b.shape()
            );
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
            print_at_level!(
                VerbosityLevel::Informational,
                "\tOutput {} has shape {:?}",
                output_name,
                a.shape()
            );
            if args.verbose >= 2 {
                a.to_file(outputs_dir, output_name)?;
            }
            node_inputs.insert(output_name.to_string(), a);
        }
        OperationResult::Multiple(res) => {
            assert_eq!(outputs.len(), res.len());
            for (output_name, res) in outputs.iter().zip(res.into_iter()) {
                print_at_level!(
                    VerbosityLevel::Informational,
                    "\tOutput {} has shape {:?}",
                    output_name,
                    res.shape()
                );
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

pub fn compare_outputs(
    expected_outputs: HashMap<String, ArrayType>,
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
                    println!("Output {} has shape {:?} as expected", name, value.shape());
                }
                if value.value_type() != data.value_type() {
                    return Err(anyhow!(
                        "Expected output {} to have type {:?} but got {:?}",
                        name,
                        value.value_type(),
                        data.value_type()
                    ));
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
                        println!(
                            "Output {} has {} values with absolute difference of more than .0001",
                            name, count
                        );
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
    Ok(())
}
