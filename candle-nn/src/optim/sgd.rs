use crate::Optimizer;
use candle::{backprop::GradStore, Var};

/// Optimizer for Stochastic Gradient Descent.
///
/// Contrary to the PyTorch implementation of SGD, this version does not support momentum.
#[derive(Debug)]
pub struct SGD {
    vars: Vec<Var>,
    learning_rate: f64,
}

impl Optimizer for SGD {
    type Config = f64;

    fn new(vars: Vec<Var>, learning_rate: f64) -> candle::Result<Self> {
        let vars = vars.into_iter().filter(|var| var.dtype().is_float()).collect();
        Ok(Self { vars, learning_rate })
    }

    fn step(&mut self, grads: &GradStore) -> candle::Result<()> {
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var) {
                var.set(&var.sub(&(grad * self.learning_rate)?)?)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr
    }
}

impl SGD {
    pub fn into_inner(self) -> Vec<Var> {
        self.vars
    }

    pub fn push(&mut self, var: &Var) {
        self.vars.push(var.clone())
    }
}
