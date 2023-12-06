mod _commonmatmul;
mod _commonpool;
pub mod add;
pub mod averagepool;
pub mod batchnormalization;
pub mod cast;
pub mod clip;
pub mod concat;
pub mod constant;
pub mod constantofshape;
pub mod conv;
pub mod div;
pub mod dropout;
pub mod exp;
pub mod flatten;
pub mod gather;
pub mod gemm;
pub mod globalaveragepool;
pub mod lrn;
pub mod matmul;
pub mod maxpool;
pub mod mul;
pub mod nonzero;
pub mod pow;
pub mod reducemean;
pub mod relu;
pub mod reshape;
pub mod shape;
pub mod slice;
pub mod softmax;
pub mod split;
pub mod sqrt;
pub mod squeeze;
pub mod sub;
pub mod sum;
pub mod tanh;
pub mod transpose;
pub mod unsqueeze;

use std::collections::HashMap;

use lazy_static::lazy_static;

use crate::common::OperationFn;

lazy_static! {
    pub static ref OPERATION_MAP: HashMap<&'static str, OperationFn> = {
        let mut m = HashMap::new();
        m.insert("Conv", conv::conv as OperationFn);
        m.insert("Clip", clip::clip as OperationFn);
        m.insert("Add", add::add as OperationFn);
        m.insert(
            "GlobalAveragePool",
            globalaveragepool::global_average_pool as OperationFn,
        );
        m.insert("Shape", shape::shape as OperationFn);
        m.insert("Constant", constant::constant as OperationFn);
        m.insert("Gather", gather::gather as OperationFn);
        m.insert("Unsqueeze", unsqueeze::unsqueeze as OperationFn);
        m.insert("Concat", concat::concat as OperationFn);
        m.insert("Reshape", reshape::reshape as OperationFn);
        m.insert("Gemm", gemm::gemm as OperationFn);
        m.insert("Relu", relu::relu as OperationFn);
        m.insert("LRN", lrn::lrn as OperationFn);
        m.insert("MaxPool", maxpool::maxpool as OperationFn);
        m.insert("Softmax", softmax::softmax as OperationFn);
        m.insert("Dropout", dropout::dropout as OperationFn);
        m.insert("Sub", sub::sub as OperationFn);
        m.insert("Div", div::div as OperationFn);
        m.insert(
            "ConstantOfShape",
            constantofshape::constantofshape as OperationFn,
        );
        m.insert("NonZero", nonzero::nonzero as OperationFn);
        m.insert("AveragePool", averagepool::averagepool as OperationFn);
        m.insert("Transpose", transpose::transpose as OperationFn);
        m.insert("Sqrt", sqrt::sqrt as OperationFn);
        m.insert("Mul", mul::mul as OperationFn);
        m.insert("Pow", pow::pow as OperationFn);
        m.insert("Squeeze", squeeze::squeeze as OperationFn);
        m.insert("Exp", exp::exp as OperationFn);
        m.insert("Tanh", tanh::tanh as OperationFn);
        m.insert("Split", split::split as OperationFn);
        m.insert("MatMul", matmul::matmul as OperationFn);
        m.insert("ReduceMean", reducemean::reducemean as OperationFn);
        m.insert("Slice", slice::slice as OperationFn);
        m.insert(
            "BatchNormalization",
            batchnormalization::batchnormalization as OperationFn,
        );
        m.insert("Cast", cast::cast as OperationFn);
        m.insert("Sum", sum::sum as OperationFn);
        m.insert("Flatten", flatten::flatten as OperationFn);
        m
    };
}
