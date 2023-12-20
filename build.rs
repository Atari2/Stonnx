use protobuf_codegen::Customize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use this in build.rs
    println!("cargo:rerun-if-changed=src/protos/onnx.proto");
    protobuf_codegen::Codegen::new()
        // Use `protoc` parser, optional.
        .protoc()
        // Use `protoc-bin-vendored` bundled protoc command, optional.
        .protoc_path(&protoc_bin_vendored::protoc_bin_path()?)
        // All inputs and imports from the inputs must reside in `includes` directories.
        .includes(["src/protos"])
        // Inputs must reside in some of include paths.
        .input("src/protos/onnx.proto")
        .cargo_out_dir("onnxparser")
        .customize(Customize::default().lite_runtime(false))
        .run_from_script();

    Ok(())
}
