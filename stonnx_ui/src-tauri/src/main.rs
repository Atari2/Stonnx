// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]


use std::collections::HashMap;

use stonnx_api::{Model, common::TensorType};
use stonnx_api::utils::OutputInfo;
use serde_json::{Value, Map};

async fn construct_json_result(result: HashMap<String, OutputInfo>) -> Result<serde_json::Value, ()> {
    let mut map = Map::new();
    for (key, value) in result {
        let mut value_map = Map::new();
        let (shape, data_type) = value.valueinfo.type_;
        value_map.insert("shape".to_string(), Value::from(data_type));
        value_map.insert("data_type".to_string(), Value::from(format!("{:?}", shape)));
        if let Some(data) = value.data {
            match data {
                TensorType::F32(data) => {
                    value_map.insert("data".to_string(), Value::String(serde_json::to_string(&data).map_err(|_| ())?))
                },
                TensorType::F64(data) => value_map.insert("data".to_string(), Value::String(serde_json::to_string(&data).map_err(|_| ())?)),
                _ => value_map.insert("data".to_string(), Value::String("data type not supported".to_string())),
            };
        } else {
            value_map.insert("data".to_string(), Value::Null);
        }
        map.insert(key, Value::from(value_map));
    }
    Ok(Value::from(map))
}

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
async fn run_model(path: String) -> Result<serde_json::Value, ()> {
    println!("Running model at path: {}", path);
    let model = Model::default().path(&path).graph(false).verbose(stonnx_api::common::VerbosityLevel::Silent).run();
    Ok(match model {
        Ok(result) => {
            construct_json_result(result).await?
        },
        Err(e) => format!("Model ran unsuccessfully: {e}").into()
    })
}


fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![run_model])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
