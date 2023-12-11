mod common;
mod executor;
mod onnxparser;
mod operators;
mod protograph;
mod utils;

use anyhow::anyhow;
use common::{BoxResult, VERBOSE};
pub use onnxparser::onnx;
use std::{path::Path, time::Instant};
pub use utils::{initialize_nodes, make_initializers, read_model, read_tensor};

use clap::Parser;

use crate::{common::{Args, FileInputs, VerbosityLevel}, executor::execute_model};

fn main() -> BoxResult<()> {
    let args = Args::parse();
    VERBOSE
        .set(VerbosityLevel::new(args.verbose as usize))
        .map_err(|_| anyhow!("Failed to set verbosity"))?;
    print_at_level!(
        VerbosityLevel::Minimal,
        "Running model: {}",
        args.model.display()
    );
    let inputspath = if args.model.is_relative() {
        Path::new("models").join(&args.model).join("inputs.json")
    } else {
        args.model.join("inputs.json")
    };
    let inputs_file = std::fs::File::open(inputspath)?;
    let mut fileinputs: FileInputs = serde_json::from_reader(inputs_file)?;
    fileinputs.extend_paths(&args.model);
    let model = read_model(Path::new(&fileinputs.modelpath))?;
    let outputs_dir = Path::new("outputs").join(&args.model);
    if VERBOSE
        .get()
        .map_or(false, |&v| v >= VerbosityLevel::Results)
    {
        std::fs::create_dir_all(&outputs_dir)?;
    }

    let start = Instant::now();
    execute_model(&model, &args, &fileinputs, &outputs_dir)?;
    let duration = start.elapsed();
    print_at_level!(
        VerbosityLevel::Minimal,
        "Time elapsed in execution is: {:?}",
        duration
    );
    Ok(())
}
