use crate::{
    common::{BoxResult, OperatorResult, TensorType},
    onnx::NodeProto,
    utils::shape_safe_product,
};
use anyhow::anyhow;

use ndarray::Ix1;

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

/// Reshape the input tensor similar to numpy.reshape <https://numpy.org/doc/stable/reference/generated/numpy.reshape.html>
///
/// First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
///
/// At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions.
///
/// A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
/// If ‘allowzero’ is set, and the new shape includes 0, the dimension will be set explicitly to zero (i.e. not taken from input tensor).
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_reshape.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__Reshape.html)
pub fn reshape(
    inputs: &[&TensorType],
    node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let data = inputs[0];
    let shape = inputs[1];
    if shape.shape().len() != 1 {
        return Err(anyhow!("shape must be 1D"));
    }
    // new_shape = np.copy(shape)
    let mut new_shape = if let TensorType::I64(shape) = shape {
        shape.view().into_dimensionality::<Ix1>()?.to_vec()
    } else {
        return Err(anyhow!("shape must be I64"));
    };
    let attrs = ReshapeAttrs::new(node);
    let datashape_array = data.shape();
    if attrs.allowzero == 0 {
        let zero_indexes = new_shape
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v == 0 { Some(i) } else { None })
            .collect::<Vec<_>>();
        if !zero_indexes.is_empty() {
            for i in zero_indexes {
                new_shape[i] = datashape_array[i] as i64;
            }
        }
    }
    let shape_tot = shape_safe_product(&new_shape);
    let data_shape_tot = shape_safe_product(data.shape()) as i64;
    if shape_tot < 0 {
        let missing_shape = data_shape_tot / -shape_tot;
        if let Some(missing_shape_index) = new_shape.iter().position(|&v| v == -1) {
            new_shape[missing_shape_index] = missing_shape;
        } else {
            return Err(anyhow!("Invalid new shape for reshape"));
        }
    }
    let new_shape = new_shape.iter().map(|&v| v as usize).collect::<Vec<_>>();
    let new_shape = ndarray::IxDyn(&new_shape);
    match data {
        TensorType::F32(data) => {
            let data = data.to_shape(new_shape)?.to_owned();
            Ok(TensorType::F32(data).into())
        }
        TensorType::I64(data) => {
            let data = data.to_shape(new_shape)?.to_owned();
            Ok(TensorType::I64(data).into())
        }
        _ => todo!("Reshape for type {} not implemented", data),
    }
}
