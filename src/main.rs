mod common;
mod executor;
mod onnxparser;
mod operators;
mod protograph;
mod utils;

use common::BoxResult;
pub use onnxparser::onnx;
use std::time::Instant;
pub use utils::{initialize_nodes, make_initializers, read_model, read_tensor};

use clap::Parser;

use crate::{
    common::{Args, VerbosityLevel},
    executor::execute_model,
};

fn main() -> BoxResult<()> {
    let args = Args::parse();
    let start = Instant::now();
    execute_model(&args)?;
    let duration = start.elapsed();
    print_at_level!(
        VerbosityLevel::Minimal,
        "Time elapsed in execution is: {:?}",
        duration
    );
    Ok(())
}
