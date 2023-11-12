use std::error::Error;

use super::super::utils::utils::ArrayType;
use super::super::onnx::AttributeProto;
use super::super::onnx::NodeProto;

enum ConvAutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

impl ConvAutoPad {
    fn from_attr(s: &AttributeProto) -> Self {
        match std::str::from_utf8(s.s()) {
            Ok("NOTSET") => Self::NotSet,
            Ok("SAME_UPPER") => Self::SameUpper,
            Ok("SAME_LOWER") => Self::SameLower,
            Ok("VALID") => Self::Valid,
            _ => Self::NotSet,
        }
    }
}

struct ConvAttributes {
    auto_pad: ConvAutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl ConvAttributes {
    fn new<'a>(node: &NodeProto, x: &ArrayType<'a>, w: &ArrayType<'a>) -> Self {
        Self {
            auto_pad: node
                .attribute
                .iter()
                .find(|a| a.name() == "auto_pad")
                .map_or(ConvAutoPad::NotSet, ConvAutoPad::from_attr),
            dilations: node
                .attribute
                .iter()
                .find(|a| a.name() == "dilations")
                .map_or_else(
                    || std::iter::repeat(1_i64).take(x.shape().len() - 2).collect(),
                    |a| a.ints.to_vec(),
                ),
            group: node
                .attribute
                .iter()
                .find(|a| a.name() == "group")
                .map_or(1, |a| a.i()),
            kernel_shape: node
                .attribute
                .iter()
                .find(|a| a.name() == "kernel_shape")
                .map_or_else(
                    || w.shape().iter().skip(2).map(|x| *x as i64).collect(),
                    |a| a.ints.to_vec(),
                ),
            pads: node
                .attribute
                .iter()
                .find(|a| a.name() == "strides")
                .map_or_else(
                    || {
                        std::iter::repeat(0_i64)
                            .take((x.shape().len() - 2) * 2)
                            .collect()
                    },
                    |a| a.ints.to_vec(),
                ),
            strides: node
                .attribute
                .iter()
                .find(|a| a.name() == "strides")
                .map_or_else(
                    || std::iter::repeat(1_i64).take(x.shape().len() - 2).collect(),
                    |a| a.ints.to_vec(),
                ),
        }
    }
}

/// CONVOLUTION
/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_conv.py
/// attributes: auto_pad, dilations, group, kernel_shape, pads, strides
/// inputs: X, W, B
///         X: input (N x C x H x W)
///             N: batch size
///             C: number of channels
///             H: height
///             W: width
///         W: weight (M x C/group x kH x kW)
///             M: number of feature maps
///             C/group: number of input channels
///             kH: kernel height
///             kW: kernel width
///         B: bias (M)
/// outputs: Y
///         Y: output
fn conv2<'a>(
    x: &ArrayType<'a>,
    w: &ArrayType<'a>,
    attrs: ConvAttributes,
) -> Result<(), Box<dyn Error>> {
    match (x, w) {
        (ArrayType::F32(x), ArrayType::F32(w)) => {
            let batch_size = x.shape()[0];
            let num_channels = x.shape()[1];
            let height = x.shape()[2];
            let width = x.shape()[3];
            let num_feature_maps = w.shape()[0];
            let num_input_channels = w.shape()[1];
            let kernel_height = w.shape()[2];
            let kernel_width = w.shape()[3];
            println!("  Convolution: batch_size: {}, num_channels: {}, height: {}, width: {}, num_feature_maps: {}, num_input_channels: {}, kernel_height: {}, kernel_width: {}", batch_size, num_channels, height, width, num_feature_maps, num_input_channels, kernel_height, kernel_width);
        }
        _ => {
            todo!("Data type not implemented");
        }
    }
    Ok(())
}
fn conv3<'a>(
    x: &ArrayType<'a>,
    w: &ArrayType<'a>,
    b: &ArrayType<'a>,
    attrs: ConvAttributes,
) -> Result<(), Box<dyn std::error::Error>> {
    match (x, w, b) {
        (ArrayType::F32(x), ArrayType::F32(w), ArrayType::F32(b)) => {
            let b = b.clone().into_dimensionality::<ndarray::Ix1>()?; // NDR: this clone is fine because we only clone the view
            let x_shape = x.shape();

            let batch_size = x.shape()[0];
            let num_channels = x.shape()[1];
            let height = x.shape()[2];
            let width = x.shape()[3];
            let num_feature_maps = w.shape()[0];
            let num_input_channels = w.shape()[1];
            let kernel_height = w.shape()[2];
            let kernel_width = w.shape()[3];
            let bias = b.shape()[0];
            println!("  Convolution: batch_size: {}, num_channels: {}, height: {}, width: {}, num_feature_maps: {}, num_input_channels: {}, kernel_height: {}, kernel_width: {}, bias_size: {}", batch_size, num_channels, height, width, num_feature_maps, num_input_channels, kernel_height, kernel_width, bias);
        }
        _ => {
            todo!("Data type not implemented");
        }
    }
    Ok(())
}

pub fn conv(inputs: &[&ArrayType], outputs: &mut Vec<&String>, node: &NodeProto) {
    assert_eq!(outputs.len(), 1);
    match inputs.len() {
        2 => {
            let (x, w) = (inputs[0], inputs[1]);
            let attributes = ConvAttributes::new(node, x, w);
            conv2(x, w, attributes).unwrap()
        }
        3 => {
            let (x, w, b) = (inputs[0], inputs[1], inputs[2]);
            let attributes = ConvAttributes::new(node, x, w);
            conv3(x, w, b, attributes).unwrap()
        }
        _ => {
            panic!("  Convolution has {} inputs", inputs.len());
        }
    }
}