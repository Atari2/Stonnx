use std::sync::{Arc, Condvar, Mutex};
type WorkFnType = Box<dyn FnOnce() + Send>;
type QueueType = Arc<(Mutex<Vec<WorkFnType>>, Condvar)>;

struct Worker {
    _thread: std::thread::JoinHandle<()>,
}

pub struct ThreadPool {
    queue: QueueType,
    _workers: Vec<Worker>,
}

impl Worker {
    pub fn new(queue: QueueType) -> Self {
        let _thread = std::thread::spawn(move || {
            let (worker_fn, cond) = &*queue;
            let mut queue = worker_fn.lock().unwrap();
            loop {
                queue = cond.wait_while(queue, |q| q.is_empty()).unwrap();
                let f = { queue.pop().unwrap() };
                f();
            }
        });
        Self { _thread }
    }
}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        let mut _workers = Vec::with_capacity(size);
        let queue = Arc::new((Mutex::new(vec![]), Condvar::new()));
        for _ in 0..size {
            _workers.push(Worker::new(queue.clone()));
        }
        Self { queue, _workers }
    }

    pub fn queue<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        {
            let mut queue = self.queue.0.lock().unwrap();
            queue.push(Box::new(f));
        }
        self.queue.1.notify_all();
    }

    pub fn wait(&mut self) {
        loop {
            {
                let queue = self.queue.0.lock().unwrap();
                if queue.is_empty() {
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}
