mod onnxparser;

use std::{path::Path, io::Read};

use onnxparser::onnx;
use walkdir::WalkDir;

fn read_model_binary(p: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(p)?;
    let mut reader = std::io::BufReader::new(file);
    let _model: onnx::ModelProto = protobuf::Message::parse_from_reader(&mut reader)?;
    // for graph in model.graph.iter() {
    //     for (i, node) in graph.node.iter().enumerate() {
    //         match node.op_type {
    //             Some(ref op_type) => match op_type.as_str() {
    //                 "MatMul" => {
    //                     println!("{}: MatMul", i);
    //                 }
    //                 "Add" => {
    //                     println!("{}: Add", i);
    //                 }
    //                 "Conv" => {
    //                     println!("{}: Conv", i);
    //                 }
    //                 _ => println!("Unimplemented op: {}", op_type),
    //             },
    //             None => println!("{}: <unknown>", i),
    //         }
    //     }
    // }
    Ok(())
}

fn read_model_text(p: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(p)?;
    let mut reader = std::io::BufReader::new(file);
    let mut buf = String::new();
    reader.read_to_string(&mut buf)?;
    let _model: onnx::ModelProto = protobuf::text_format::parse_from_str(&buf)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for entry in WalkDir::new("examples") {
        let entry = entry?;
        if entry.path().extension().map_or(false, |e| e == "onnx") {
            match read_model_binary(entry.path()) {
                Ok(_) => println!("{}: OK", entry.path().display()),
                Err(e) => {
                    println!("Error during binary parsing of {}: {}, trying with text", entry.path().display(), e);
                    match read_model_text(entry.path()) {
                        Ok(_) => println!("{}: OK", entry.path().display()),
                        Err(e) => println!("Error during text parsing of {}: {}", entry.path().display(), e),
                    }
                },
            }
        }
    }

    Ok(())
}
