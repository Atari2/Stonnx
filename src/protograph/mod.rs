use anyhow::anyhow;
use petgraph::dot::{Config, Dot};
use petgraph::graphmap::GraphMap;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::{collections::HashMap, path::Path};

use crate::{common::BoxResult, onnx::GraphProto};

#[derive(
    Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Serialize, Deserialize, std::hash::Hash,
)]
/// The type of node, either an input/output node, or an operator node.
enum NodeType {
    InputOutput,
    Operator,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, std::hash::Hash,
)]
/// A node in the graph, with a name and a type.
struct Node<'a> {
    name: &'a str,
    node_type: NodeType,
}

impl<'a> Display for Node<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.node_type {
            NodeType::InputOutput => write!(f, "{}", self.name),
            NodeType::Operator => write!(f, "{}", self.name),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum GraphOutputType {
    Json,
    Dot,
}

/// Given a graph, and the model's path, it will create a graph.{json|dot} file in the same directory.
///
/// This file can be used to visualize the graph using a custom made tool called [ONNXGraphLayout](https://github.com/Atari2/ONNXGraphLayout).
///
/// The format of the graph is (by default) simple JSON and as such can be easily parsed by other tools.
///
/// The top level JSON object is has 2 keys, "nodes" and "edges".
/// - "nodes" is an array of objects, each object has a "name" and a "node_type" key. Both are strings, and the node_type can be either "InputOutput" or "Operator".
/// - "edges" is an array of arrays, each array has, in order, 1 number that represents the index of the source node, 1 number that represents the index of the target node and a string that represents the name of the edge.
///   The index of the source/target node is the 0-based index of the node in the "nodes" array.
///
/// The format can also be changed to dot, which is a format used by GraphViz.
pub fn build_graph_from_proto(
    proto: &GraphProto,
    modelpath: &Path,
    output_type: GraphOutputType,
) -> BoxResult<()> {
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
    if output_type == GraphOutputType::Json {
        let graph_string = serde_json::to_string_pretty(&graph)?;
        std::fs::write(
            modelpath
                .parent()
                .ok_or_else(|| anyhow!("Modelpath has no parent"))?
                .join("graph.json"),
            graph_string,
        )?;
        return Ok(());
    } else {
        let dot = Dot::with_config(&graph, &[Config::EdgeNoLabel]);
        std::fs::write(
            modelpath
                .parent()
                .ok_or_else(|| anyhow!("Modelpath has no parent"))?
                .join("graph.dot"),
            format!("{}", dot),
        )?;
    }
    Ok(())
}
