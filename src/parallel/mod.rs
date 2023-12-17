use std::sync::{Arc, Condvar, Mutex};
type WorkFnType = Box<dyn FnOnce() + Send>;
type QueueType = Arc<(Mutex<Vec<WorkerMessage>>, Condvar)>;

enum WorkerMessage {
    Work(WorkFnType),
    Stop,
}

impl WorkerMessage {
    fn is_stop(&self) -> bool {
        matches!(self, Self::Stop)
    }
    fn get_fn(self) -> WorkFnType {
        match self {
            Self::Work(f) => f,
            _ => panic!("WorkerMessage::get_fn called on non-Work variant"),
        }
    }
}

struct Worker {
    thread: std::thread::JoinHandle<()>,
}

pub struct ThreadPool {
    queue: QueueType,
    workers: Vec<Worker>,
}

impl Worker {
    pub fn new(queue: QueueType) -> Self {
        let thread = std::thread::spawn(move || {
            let (worker_fn, cond) = &*queue;
            let mut queue = worker_fn.lock().expect("Locking worker queue failed");
            loop {
                queue = cond
                    .wait_while(queue, |q| q.is_empty())
                    .expect("Waiting on worker queue failed");
                if queue.iter().any(|x| x.is_stop()) {
                    break;
                }
                let f = { queue.pop().expect("Worker queue is empty").get_fn() };
                f();
            }
        });
        Self { thread }
    }
}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        let mut workers = Vec::with_capacity(size);
        let queue = Arc::new((Mutex::new(vec![]), Condvar::new()));
        for _ in 0..size {
            workers.push(Worker::new(queue.clone()));
        }
        Self { queue, workers }
    }

    pub fn queue<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        {
            let mut queue = self.queue.0.lock().expect("Locking worker queue failed");
            queue.push(WorkerMessage::Work(Box::new(f)));
        }
        self.queue.1.notify_all();
    }

    pub fn wait(&mut self) {
        loop {
            let queue = self.queue.0.lock().expect("Locking worker queue failed");
            if queue.is_empty() {
                break;
            }
        }
    }

    fn stop(&mut self) {
        {
            let mut queue = self.queue.0.lock().expect("Locking worker queue failed");
            queue.push(WorkerMessage::Stop);
        }
        self.queue.1.notify_all();
        for worker in self.workers.drain(..) {
            worker.thread.join().expect("Joining worker thread failed");
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.stop();
    }
}
