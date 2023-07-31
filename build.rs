use std::env;

use protobuf_codegen::Customize;

fn main() {
    // Use this in build.rs
    env::set_var("OUT_DIR", "src");
    println!("cargo:rerun-if-changed=src/protos/onnx.proto");
    protobuf_codegen::Codegen::new()
        // Use `protoc` parser, optional.
        .protoc()
        // Use `protoc-bin-vendored` bundled protoc command, optional.
        .protoc_path(&protoc_bin_vendored::protoc_bin_path().unwrap())
        // All inputs and imports from the inputs must reside in `includes` directories.
        .includes(["src/protos"])
        // Inputs must reside in some of include paths.
        .input("src/protos/onnx.proto")
        // Specify output directory relative to Cargo output directory.
        .cargo_out_dir("onnxparser")
        .customize(Customize::default().lite_runtime(false))
        .run_from_script();
}
