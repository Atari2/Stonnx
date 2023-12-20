use curl::easy::Easy;
use std::sync::{Arc, Mutex};
use std::{env, io::Write, path::PathBuf};
use zip_extensions::zip_extract;

use protobuf_codegen::Customize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use this in build.rs

    let in_ci = env::var("CI").is_ok_and(|x| x == "true");
    if !in_ci {
        // if not on CI, download models
        if std::path::Path::new("models").exists() {
            println!("models already downloaded");
        } else {
            println!("cargo:warning=Downloading ONNX models, this may take a while");
            println!("downloading models");
            std::fs::create_dir("models")?;
            let file = std::fs::File::create("models/models.zip")?;
            let writer = Arc::new(Mutex::new(std::io::BufWriter::new(file)));
            {
                let mut easy = Easy::new();
                easy.url("https://www.atarismwc.com/models.zip")?;
                let writer_clone = Arc::clone(&writer);
                easy.write_function(move |data| {
                    let mut writer = writer_clone.lock().expect("Failed to lock writer");
                    writer.write_all(data).expect("Failed to write data");
                    Ok(data.len())
                })?;
                easy.perform()?;
            }
            let mut writer = Arc::into_inner(writer)
                .expect("Failed to unwrap Arc")
                .into_inner()?;
            writer.flush()?;
            let archive_file: PathBuf = "models/models.zip".into();
            let target_dir: PathBuf = "models".into();
            zip_extract(&archive_file, &target_dir)?;
            std::fs::remove_file("models/models.zip")?;
        }
    } else {
        println!("cargo:warning=Not downloading models because we are on CI or building for package registry");
    }

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
