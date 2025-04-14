use crate::{bail, tweaks::module::DynModule, Result, Tensor};
use rayon::{prelude::*, ThreadPool};

pub struct ParallelStage<F> {
    //pool: ThreadPool,
    stages: Vec<DynModule>,
    buffer: Vec<Tensor>,
    end_stage: Vec<F>,
}

impl<F> ParallelStage<F> {
    pub fn new(parallelism: Option<usize>) -> Self {
        Self {
            /*pool: rayon::ThreadPoolBuilder::new()
            .num_threads(parallelism.unwrap_or(rayon::current_num_threads()))
            .build()
            .unwrap(),*/
            stages: Vec::new(),
            buffer: Vec::new(),
            end_stage: Vec::new(),
        }
    }

    pub fn add(&mut self, stage: impl Into<DynModule>) {
        self.stages.push(stage.into());
    }

    pub fn add_fn(&mut self, stage: fn(&Tensor) -> Result<Tensor>) {
        self.stages.push(stage.into());
    }

    #[inline]
    fn with_pool<R: Send>(&self, closure: impl Send + FnOnce(&ThreadPool) -> R) -> R {
        // self.pool.install(|| closure(&self.pool))
        closure(&rayon::ThreadPoolBuilder::new().build().unwrap())
    }

    /// same as `Sequential::forward_t`
    /* pub fn forward_t(&self, xs: impl Send + IntoParallelIterator<Item = Tensor>, train: bool) -> Result<Vec<Tensor>> {
        self.with_pool(|_| {
            xs.into_par_iter()
                .map(|mut res| {
                    for module in &self.stages {
                        res = module.forward_t(&res, train)?;
                    }
                    Ok(res)
                })
                .collect()
        })
    }*/

    pub fn forward_par<R>(&mut self, xs: Tensor, end_stage: F) -> Result<Option<R>>
    where
        F: Send + Sync + FnOnce(&Tensor) -> Result<R>,
    {
        self.buffer.insert(0, xs);
        self.end_stage.insert(0, end_stage);
        let mut end_stage = None;
        if self.buffer.len() > self.stages.len() {
            self.buffer.pop();
            end_stage = self.end_stage.pop();
        }
        let job_count = self.buffer.len();
        let mut work = WorkSlot::<F, R> {
            buffer: vec![],
            end_res: None,
            _phantom: std::marker::PhantomData,
        };
        for _ in 0..job_count {
            work.buffer.push(None);
        }
        let mut jobs = Vec::<Box<dyn FnOnce() + Send + Sync>>::new();
        for i in 0..job_count {
            let storage = SendMutPtr(&mut work.buffer[i]);
            let module = SendPtr(&self.stages[i]);
            let xs = SendPtr(&self.buffer[i]);
            jobs.push(Box::new(move || {
                let (storage, module, xs) = (storage, module, xs);
                let (module, xs) = unsafe { (&*module.0, &*xs.0) };
                let mut tensor = Some(module.forward_t(xs, true));
                unsafe {
                    storage.0.swap(&mut tensor);
                }
            }));
        }
        if let Some(end_stage) = end_stage {
            // also perform backward pass
            let last = SendPtr(&work.buffer[job_count - 1]);
            let grads = SendMutPtr(&mut work.end_res);
            jobs.push(Box::new(move || {
                let (last, grads) = (last, grads);
                let last = unsafe { &*last.0 };
                while last.is_none() {
                    rayon::yield_now();
                }
                if let Some(Ok(tensor)) = last {
                    let grad = end_stage(tensor);
                    unsafe { *grads.0 = Some(grad) }
                } else {
                    let error = Err(crate::Error::msg("Error before backward pass"));
                    unsafe { *grads.0 = Some(error) }
                }
            }));
        }
        jobs.into_par_iter().for_each(|job| job());
        self.buffer = work
            .buffer
            .into_iter()
            .map(|ptr| {
                if ptr.is_none() {
                    bail!("null pointer")
                }
                unsafe { ptr.unwrap() }
            })
            .collect::<Result<Vec<_>>>()?;
        let grad = work.end_res;
        grad.transpose()
    }

    pub fn drive<R>(&mut self)->Result<Vec<R>>
    where
        F: Send + Sync + FnOnce(&Tensor) -> Result<R> {

        let mut work = WorkSlot::<F, R> {
            buffer: vec![],
            end_res: None,
            _phantom: std::marker::PhantomData,
        };
        let jobs = self.buffer.len();
        let mut res = Vec::with_capacity(jobs);
        work.buffer.extend(self.buffer.drain(..).map(|x| Some(Ok(x))));
        for _ in 0..(self.stages.len() - jobs) {
            work.buffer.push(None);
        }

        todo!("drive all remaining jobs");

        Ok(res)
    }
}

struct WorkSlot<F: FnOnce(&Tensor) -> Result<R>, R> {
    buffer: Vec<Option<Result<Tensor>>>,
    end_res: Option<Result<R>>,
    _phantom: std::marker::PhantomData<F>,
}
struct SendPtr<T>(*const T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
struct SendMutPtr<T>(*mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device};
    #[test]
    fn test() {
        let mut stage = ParallelStage::new(None);
        stage.add_fn(|tensor| Ok(tensor.clone()));
        stage.add_fn(|tensor| Ok(tensor.clone()));
        let backward = |t: &Tensor| t.backward();
        let first = stage
            .forward_par(
                Tensor::ones(vec![1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap(),
                backward,
            )
            .unwrap();
        assert!(first.is_none());

        let second = stage
            .forward_par(
                Tensor::ones(vec![1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap(),
                backward,
            )
            .unwrap();
        assert!(second.is_none());

        let third = stage
            .forward_par(
                Tensor::ones(vec![1, 2, 3, 4], DType::F32, &Device::Cpu).unwrap(),
                backward,
            )
            .unwrap();
        assert!(third.is_some());
    }
}
