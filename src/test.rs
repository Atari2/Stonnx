#[cfg(test)]

mod tests {
    use ndarray::{ArrayD, IxDyn};

    use crate::{common::TensorType, onnx::NodeProto, operators::add};

    #[test]
    fn test_add() {
        let node = NodeProto::default();
        let a = TensorType::F32(ArrayD::<f32>::from_elem(IxDyn(&[2, 2]), 0f32));
        let b = TensorType::F32(ArrayD::<f32>::from_elem(IxDyn(&[2, 2]), 1f32));
        let c = add::add(&[&a, &b], &node, 6, 1);
        assert!(c.is_ok());
        let c = c.unwrap();
        assert_eq!(c.result.len(), 1);
        let c = &c.result[0];
        assert!(c.allclose(
            &TensorType::F32(ArrayD::<f32>::from_elem(IxDyn(&[2, 2]), 1f32)),
            None,
            None,
            None
        ));
    }
}
