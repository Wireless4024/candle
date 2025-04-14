use crate::Optimizer;
use candle::{backprop::GradStore, Tensor};

pub trait LRScheduler {
    type Optimizer: Optimizer;
    type Config;
    fn new(optimizer: Self::Optimizer) -> Self
    where
        Self::Config: Default,
        Self: Sized,
    {
        Self::new_with_optimizer(Self::Config::default(), optimizer)
    }
    fn new_with_optimizer(config: impl Into<Self::Config>, optimizer: Self::Optimizer) -> Self;
    fn step(&mut self, grads: &GradStore) -> candle::Result<()>;
    fn backward_step(&mut self, loss: &Tensor) -> candle::Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }
    fn learning_rate(&self) -> f64;
    fn set_learning_rate(&mut self, lr: f64);
    fn is_finished(&self) -> bool {
        false
    }
    /// called after each batch, can be used to track the current loss
    fn current_loss(&mut self, loss: f64) {}
    /// called after each epoch end
    fn next_epoch(&mut self) -> candle::Result<()> {
        Ok(())
    }
    fn into_inner(self) -> Self::Optimizer;
    fn forward_to<LR: LRScheduler<Optimizer = Self::Optimizer>>(self, cfg: LR::Config) -> LR
    where
        Self: Sized,
    {
        LR::new_with_optimizer(cfg, self.into_inner())
    }
}

pub struct NoopLRScheduler<O: Optimizer> {
    optimizer: O,
}

impl<O: Optimizer> LRScheduler for NoopLRScheduler<O> {
    type Optimizer = O;
    type Config = ();

    fn new_with_optimizer(_: impl Into<Self::Config>, optimizer: Self::Optimizer) -> Self {
        Self { optimizer }
    }

    fn step(&mut self, grads: &GradStore) -> candle::Result<()> {
        self.optimizer.step(grads)
    }

    fn learning_rate(&self) -> f64 {
        self.optimizer.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.optimizer.set_learning_rate(lr)
    }

    fn into_inner(self) -> Self::Optimizer {
        self.optimizer
    }
}
