#[cfg(test)]

mod tests {
    use ndarray::{ArrayD, IxDyn};

    use crate::{common::TensorType, onnx::NodeProto, operators::add, parallel};

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

    #[test]
    fn parallel_test() {
        let mut pool = parallel::ThreadPool::new(4);
        for _ in 0..100 {
            pool.queue(|| {
                std::thread::sleep(std::time::Duration::from_millis(
                    rand::random::<u64>() % 100,
                ));
                println!("Hello from thread {:?}!", std::thread::current().id());
            });
        }
    }
}
