mod onnxparser;
use std::io::Read;

use onnxparser::onnx;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open("examples/test.onnx")?;
    let mut str = String::new();
    file.read_to_string(&mut str)?;
    let model = protobuf::text_format::parse_from_str::<onnx::ModelProto>(&str)?;
    for graph in model.graph.iter() {
        for (i, node) in graph.node.iter().enumerate() {
            println!("node: {}, op_type: {}", i, node.op_type.as_ref().unwrap());
            node.input.iter().for_each(|input| println!("\tinput: {}", input));
            node.output.iter().for_each(|output| println!("\toutput: {}", output));
        }
    }
    Ok(())
}