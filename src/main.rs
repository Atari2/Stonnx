mod common;
mod executor;
mod onnxparser;
mod operators;
mod parallel;
mod protograph;
mod test;
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
    // let mut pool = ThreadPool::new(4);
    // let arc = Arc::new(Mutex::new(0));
    // for _ in 0..10 {
    //     let arc = arc.clone();
    //     pool.execute(move || {
    //         {
    //             let mut lock = arc.lock().unwrap();
    //             *lock += 1;
    //         }
    //         std::thread::sleep(std::time::Duration::from_millis(200));
    //     });
    // }
    // pool.wait();
    // let results = match Arc::into_inner(arc) {
    //     Some(r) => r.into_inner()?,
    //     // fails here for some reason?
    //     // it is very weird, because the pool.wait() call before
    //     // should make sure that all threads are done and as such all the refs are dropped except for the main one
    //     None => return Err(anyhow!("Arc was poisoned!")),
    // };
    // println!("Results: {}", results);
    // Ok(())
}
