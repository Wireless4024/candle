//! Various optimization algorithms.

pub mod adamw;
pub mod gradclip;
pub mod lion;
pub mod muon;
pub mod nadamw;
pub mod scheduler;
pub mod sgd;
pub mod sophia;

use candle::{backprop::GradStore, Result, Tensor, Var};

use crate::tweaks::SafeTensorStorage;
pub use adamw::*;

/// The interface optimizers should implement.
#[allow(unused_variables)]
pub trait Optimizer: Sized {
    type Config: Sized;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self>;

    fn new2(vars: Vec<candle::tweaks::NamedVar>, params: Self::Config, vb: crate::VarBuilder) -> Result<Self> {
        Self::new(vars.into_iter().map(|v| v.var().clone()).collect(), params)
    }

    fn step(&mut self, grads: &GradStore) -> Result<()>;

    fn learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, lr: f64);

    fn empty(config: Self::Config) -> Result<Self> {
        Self::new(vec![], config)
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    fn from_slice(vars: &[&Var], config: Self::Config) -> Result<Self> {
        let vars: Vec<_> = vars.iter().map(|&v| v.clone()).collect();
        Self::new(vars, config)
    }

    fn zeroed(&mut self) -> Result<()> {
        Ok(())
    }

    fn load_state(&mut self, storage: &SafeTensorStorage) -> Result<()> {
        Ok(())
    }

    fn save_state(&self, storage: &SafeTensorStorage) {}

    fn boxed_dyn(self) -> Box<dyn DynOptimizer>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

pub trait OptimizerExt {
    type StepOpt;
    type Var;
    fn vars(&self) -> &[Self::Var];
    fn step_opt(&self) -> Self::StepOpt;
    fn step_once(&self, gradient: &Tensor, var: &Self::Var, opt: &Self::StepOpt) -> Result<()>;
}

pub trait DynOptimizer {
    fn step(&mut self, grads: &GradStore) -> Result<()>;
    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }
    fn load_state(&mut self, storage: &SafeTensorStorage) -> Result<()>;
    fn save_state(&self, storage: &SafeTensorStorage);
    fn learning_rate(&self) -> f64;
    fn set_learning_rate(&mut self, lr: f64);
}

impl<T: Optimizer> DynOptimizer for T {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        Optimizer::step(self, grads)
    }

    fn load_state(&mut self, storage: &SafeTensorStorage) -> Result<()> {
        Optimizer::load_state(self, storage)
    }

    fn save_state(&self, storage: &SafeTensorStorage) {
        Optimizer::save_state(self, storage);
    }

    fn learning_rate(&self) -> f64 {
        Optimizer::learning_rate(self)
    }

    fn set_learning_rate(&mut self, lr: f64) {
        Optimizer::set_learning_rate(self, lr)
    }
}

const _DYN_OPT_COMPAT: Option<Box<dyn DynOptimizer>> = None;
