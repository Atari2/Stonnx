use anyhow::anyhow;
use petgraph::graphmap::GraphMap;
use std::{collections::HashMap, path::Path};

use crate::{common::BoxResult, onnx::GraphProto};

pub fn build_graph_from_proto(proto: &GraphProto, modelpath: &Path) -> BoxResult<()> {
    let mut graph = GraphMap::<&str, &str, petgraph::Directed>::new();
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
        let op_node = graph.add_node(node_names[i].as_str());
        let mut inputs = vec![];
        let mut outputs = vec![];
        for input in node.input.iter() {
            if input.is_empty() {
                eprintln!(
                    "Empty input for node {}",
                    node.name.as_ref().map_or("UNKNOWN", |x| x.as_str())
                );
            }
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
