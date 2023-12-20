use curl::easy::Easy;
use std::{env, io::Write, path::PathBuf};
use zip_extensions::zip_extract;

use protobuf_codegen::Customize;

fn main() {
    // Use this in build.rs

    // models contained in models.zip
    let modelnames = [
        "vgg19-7",
        "GPT2",
        "zfnet512-12",
        "caffenet-12",
        "bvlcalexnet-12",
    ];
    let in_ci = env::var("CI").is_ok_and(|x| x == "true");
    if !in_ci {
        // if not on CI, download models
        if modelnames
            .iter()
            .all(|modelname| std::path::Path::new("models").join(modelname).exists())
        {
            println!("models already downloaded");
        } else {
            println!("downloading models");
            let mut easy = Easy::new();
            easy.url("https://www.atarismwc.com/models.zip").unwrap();
            let file = std::fs::File::create("models/models.zip").unwrap();
            let mut writer = std::io::BufWriter::new(file);
            easy.write_function(move |data| {
                writer.write_all(data).unwrap();
                Ok(data.len())
            })
            .unwrap();
            easy.perform().unwrap();
            let archive_file: PathBuf = "models/models.zip".into();
            let target_dir: PathBuf = "models".into();
            zip_extract(&archive_file, &target_dir).unwrap();
            std::fs::remove_file("models/models.zip").unwrap();
        }
    }

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
