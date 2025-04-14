//! Sequential Layer
//!
//! A sequential layer used to chain multiple layers and closures.

use candle::{tweaks::module::DynModule, ModuleT, Result, Tensor};
use std::{fmt::Debug};

/// A sequential layer combining multiple other layers.
pub struct Sequential {
    layers: Vec<DynModule>,
}

impl Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sequential")
    }
}

/// Creates a new empty sequential layer.
pub fn seq() -> Sequential {
    Sequential { layers: vec![] }
}

impl Sequential {
    /// The number of sub-layers embedded in this layer.
    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }

    /// Returns true if this layer does not have any sub-layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl ModuleT for Sequential {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        if self.layers.is_empty() {
            return Ok(xs.clone());
        }
        let mut tensors = self.forward_all_t(xs, train)?;
        Ok(tensors.pop().unwrap())
    }
}

impl Sequential {
    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Into<DynModule>>(mut self, layer: M) -> Self {
        self.layers.push(layer.into());
        self
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) -> Result<Tensor> + Send + Sync,
    {
        self.add(super::func(f))
    }

    pub fn insert<M: Into<DynModule> + 'static>(&mut self, index: usize, layer: M) {
        self.layers.insert(index, layer.into());
    }

    pub fn insert_fn<F>(&mut self, index: usize, f: F)
    where
        F: 'static + Fn(&Tensor) -> Result<Tensor> + Send + Sync,
    {
        self.insert(index, super::func(f));
    }

    #[inline]
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_t(xs, false)
    }

    /// Applies the forward pass and returns the output for each layer.
    #[inline]
    pub fn forward_all(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        self.forward_all_t(xs, false)
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all_t(&self, xs: &Tensor, training: bool) -> Result<Vec<Tensor>> {
        if self.layers.is_empty() {
            return Ok(vec![xs.clone()]);
        }
        let mut vec = Vec::with_capacity(self.layers.len());
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward_t(&xs, training)?;
            vec.push(xs.clone())
        }
        Ok(vec)
    }
/*
    /// forward all items in parallel
    pub fn forward_all_par<T: AsRef<Tensor>>(
        &self,
        items: impl rayon::iter::IntoParallelIterator<Item = T>,
        train: bool,
    ) -> Result<Vec<Tensor>> {
        use rayon::prelude::*;
        items
            .into_par_iter()
            .map(|x| self.forward_t(x.as_ref(), train))
            .collect()
    }*/
}
