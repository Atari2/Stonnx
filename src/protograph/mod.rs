use anyhow::anyhow;
use petgraph::graphmap::GraphMap;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};

use crate::{common::BoxResult, onnx::GraphProto};

#[derive(
    Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Serialize, Deserialize, std::hash::Hash,
)]
enum NodeType {
    InputOutput,
    Operator,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, std::hash::Hash,
)]
struct Node<'a> {
    name: &'a str,
    node_type: NodeType,
}

pub fn build_graph_from_proto(proto: &GraphProto, modelpath: &Path) -> BoxResult<()> {
    let mut graph = GraphMap::<Node<'_>, &str, petgraph::Directed>::new();
    let mut count_map = HashMap::<&str, usize>::new();
    let mut node_names = vec![];
    for input in proto.node.iter() {
        if input.op_type.is_none() {
            node_names.push(format!("UNNAMED_{}", node_names.len()));
        } else if let Some(ref name) = input.op_type {
            if name.is_empty() {
                node_names.push(format!("UNNAMED_{}", node_names.len()));
            } else {
                let count = count_map
                    .entry(name.as_str())
                    .and_modify(|x| *x += 1)
                    .or_insert(0);
                node_names.push(format!("{}_{}", name, count));
            }
        }
    }
    for (i, node) in proto.node.iter().enumerate() {
        let op_node = graph.add_node(Node {
            name: node_names[i].as_str(),
            node_type: NodeType::Operator,
        });
        let mut inputs = vec![];
        let mut outputs = vec![];
        for input in node.input.iter() {
            if input.is_empty() {
                eprintln!(
                    "Empty input for node {}",
                    node.name.as_ref().map_or("UNKNOWN", |x| x.as_str())
                );
            }
            let input = Node {
                name: input,
                node_type: NodeType::InputOutput,
            };
            if !graph.contains_node(input) {
                inputs.push(graph.add_node(input));
            } else {
                inputs.push(graph.nodes().find(|x| *x == input).unwrap());
            }
        }
        for output in node.output.iter() {
            if output.is_empty() {
                eprintln!(
                    "Empty output for node {}",
                    node.name.as_ref().map_or("UNKNOWN", |x| x.as_str())
                );
            }
            let output = Node {
                name: output,
                node_type: NodeType::InputOutput,
            };
            if !graph.contains_node(output) {
                outputs.push(graph.add_node(output));
            } else {
                outputs.push(graph.nodes().find(|x| *x == output).unwrap());
            }
        }
        for input in inputs {
            graph.add_edge(input, op_node, "input");
        }
        for output in outputs {
            graph.add_edge(op_node, output, "output");
        }
    }
    let graph_string = serde_json::to_string_pretty(&graph)?;
    std::fs::write(
        modelpath
            .parent()
            .ok_or_else(|| anyhow!("Modelpath has no parent"))?
            .join("graph.json"),
        graph_string,
    )?;
    Ok(())
}
