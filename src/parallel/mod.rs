use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// Type alias for a function that can be queued to a thread pool.
type WorkFnType = Box<dyn FnOnce() + Send>;
/// Type alias for the queue used by the thread pool.
type QueueType = Arc<(Mutex<Vec<WorkerMessage>>, Condvar)>;
/// Type alias for the queue state used by the thread pool.
type QueueStateType = Arc<(Mutex<usize>, Condvar)>;

/// Enum representing the messages that can be sent to a worker thread.
enum WorkerMessage {
    /// A function to be executed by the worker thread.
    Work(WorkFnType),
    /// A message to stop the worker thread.
    Stop,
}

impl WorkerMessage {
    /// Returns true if the message is a stop message.
    fn is_stop(&self) -> bool {
        matches!(self, Self::Stop)
    }
    /// Returns the function contained in the message.
    /// Panics if the message is not a Work variant.
    fn get_fn(self) -> WorkFnType {
        match self {
            Self::Work(f) => f,
            _ => panic!("WorkerMessage::get_fn called on non-Work variant"),
        }
    }
}

/// Struct representing a worker thread.
struct Worker {
    thread: std::thread::JoinHandle<()>,
}

/// Struct representing a thread pool.
pub struct ThreadPool {
    queue: QueueType,
    workers: Vec<Worker>,
    queuestate: QueueStateType,
}

/// A function that tries to lock a mutex and yields if it fails.
/// Panics if the mutex is poisoned.
fn ylock<T>(mutex: &Mutex<T>) -> MutexGuard<T> {
    loop {
        match mutex.try_lock() {
            Ok(guard) => break guard,
            Err(std::sync::TryLockError::WouldBlock) => {
                std::thread::yield_now();
            }
            Err(std::sync::TryLockError::Poisoned(e)) => {
                panic!("Locking mutex failed with error: {:?}", e);
            }
        }
    }
}

impl Worker {
    /// Creates a new worker thread, which will execute functions from the given queue.
    pub fn new(queue: QueueType, queuestate: QueueStateType) -> Self {
        let thread = std::thread::spawn(move || {
            let (worker_fn, cond) = &*queue;
            let mut queue = ylock(worker_fn);
            loop {
                let f = {
                    queue = cond
                        .wait_while(queue, |q| q.is_empty())
                        .expect("Waiting on worker queue failed");
                    if queue.iter().any(|x| x.is_stop()) {
                        break;
                    }
                    if let Some(work_item) = queue.pop() {
                        work_item.get_fn()
                    } else {
                        continue;
                    }
                };
                drop(queue);
                f();
                let (worker_state, worker_cond) = &*queuestate;
                {
                    let mut state = ylock(worker_state);
                    *state -= 1;
                    worker_cond.notify_all();
                }
                queue = ylock(worker_fn);
            }
        });
        Self { thread }
    }
}

impl ThreadPool {
    /// Creates a new thread pool with the given number of worker threads.
    pub fn new(size: usize) -> Self {
        let mut workers = Vec::with_capacity(size);
        let queue = Arc::new((Mutex::new(vec![]), Condvar::new()));
        let queuestate = Arc::new((Mutex::new(0), Condvar::new()));
        for _ in 0..size {
            workers.push(Worker::new(queue.clone(), queuestate.clone()));
        }
        Self {
            queue,
            workers,
            queuestate,
        }
    }

    /// Queues a function to be executed by the thread pool.
    pub fn queue<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        {
            let mut queue = ylock(&self.queue.0);
            {
                let mut queuestate = ylock(&self.queuestate.0);
                *queuestate += 1;
            }
            queue.push(WorkerMessage::Work(Box::new(f)));
        }
        self.queue.1.notify_all();
    }

    /// Waits for all queued functions to be executed.
    pub fn wait(&mut self) {
        let (queuestate, cond) = &*self.queuestate;
        let mut state = ylock(queuestate);
        while *state > 0 {
            state = cond
                .wait_while(state, |s| *s > 0)
                .expect("Waiting on worker queue state failed");
        }
    }

    /// Stops all worker threads.
    fn stop(&mut self) {
        {
            let mut queue = ylock(&self.queue.0);
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
