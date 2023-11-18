// TODO: remove this when operator is implemented
use crate::{
    onnx::NodeProto,
    utils::{shape_safe_product, ArrayType, BoxResult, OperationResult},
};

use ndarray::IxDyn;

const _OPSET_VERSIONS: [i64; 5] = [1, 5, 13, 14, 19];

#[derive(Debug)]
struct ReshapeAttrs {
    allowzero: i64,
}

impl ReshapeAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            allowzero: node
                .attribute
                .iter()
                .find(|a| a.name() == "allowzero")
                .map_or(0, |a| a.i.unwrap_or(0)),
        }
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_reshape.py
/// https://onnx.ai/onnx/operators/onnx__Reshape.html
pub fn reshape(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let data = inputs[0];
    let shape = inputs[1];
    if shape.shape().len() != 1 {
        return Err("shape must be 1D".into());
    }
    // new_shape = np.copy(shape)
    let shape = if let ArrayType::I64(shape) = shape {
        shape
    } else {
        return Err("shape must be I64".into());
    };
    let mut new_shape = shape.clone();
    let attrs = ReshapeAttrs::new(node);
    let datashape_array =
        ndarray::ArrayD::from_shape_vec(IxDyn(&[data.shape().len()]), data.shape().to_vec())?
            .into_dyn();
    if attrs.allowzero == 0 {
        let zero_indexes = shape
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v == 0 { Some(i) } else { None })
            .collect::<Vec<_>>();
        if !zero_indexes.is_empty() {
            new_shape[zero_indexes.as_slice()] = datashape_array[zero_indexes.as_slice()] as i64;
        }
    }
    let shape_tot = shape_safe_product(&new_shape);
    let data_shape_tot = shape_safe_product(data.shape()) as i64;
    if shape_tot < 0 {
        let missing_shape = data_shape_tot / -shape_tot;
        if let Some(missing_shape_index) = new_shape.iter().position(|&v| v == -1) {
            new_shape[missing_shape_index] = missing_shape;
        } else {
            return Err("Invalid new shape for reshape".into());
        }
    }
    let new_shape = new_shape.iter().map(|&v| v as usize).collect::<Vec<_>>();
    let new_shape = ndarray::IxDyn(&new_shape);
    match data {
        ArrayType::F32(data) => {
            let data = data.to_shape(new_shape)?.to_owned();
            Ok(ArrayType::F32(data).into())
        }
        ArrayType::I64(data) => {
            let data = data.to_shape(new_shape)?.to_owned();
            Ok(ArrayType::I64(data).into())
        }
        _ => todo!("Reshape for type {} not implemented", data),
    }
}
