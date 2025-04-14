use crate::tweaks::{ModuleRegistry, SerializableModule};
use candle::{tweaks::module::DynModule, ModuleT, Tensor};
use serde_json::Value;
use std::borrow::Cow;

pub struct SerializableSequential {
    layers: Vec<DynModule>,
    cfg: Vec<Value>,
}

impl SerializableSequential {
    pub fn new() -> Self {
        Self {
            layers: vec![],
            cfg: vec![],
        }
    }

    pub fn add<M: SerializableModule>(mut self, layer: M, name: impl ToString) -> Self {
        self.cfg.push(layer.config_value(name));
        self.layers.push(DynModule::new(layer));
        self
    }

    pub fn insert<M: SerializableModule>(&mut self, index: usize, layer: M, name: impl ToString) {
        self.cfg.insert(index, layer.config_value(name));
        self.layers.insert(index, DynModule::new(layer));
    }

    pub fn forward_all(&self, xs: &Tensor) -> candle::Result<Vec<Tensor>> {
        self.forward_all_t(xs, false)
    }

    pub fn forward_all_t(&self, xs: &Tensor, train: bool) -> candle::Result<Vec<Tensor>> {
        if self.layers.is_empty() {
            return Ok(vec![xs.clone()]);
        }
        let mut outs = vec![];
        for layer in &self.layers {
            outs.push(layer.forward_t(xs, train)?);
        }
        Ok(outs)
    }
}

impl ModuleT for SerializableSequential {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle::Result<Tensor> {
        self.forward_all_t(xs, train).map(|x| x.into_iter().last().unwrap())
    }
}

impl SerializableModule for SerializableSequential {
    type Config = Vec<Value>;

    fn load(config: Self::Config, registry: &ModuleRegistry, vb: crate::VarBuilder) -> Result<Self, candle::Error> {
        let mut layers = vec![];
        for config in &config {
            let module = registry.load(
                serde_json::from_value(config.clone()).map_err(candle::Error::wrap)?,
                &vb,
            )?;
            layers.push(module);
        }
        Ok(Self { layers, cfg: config })
    }

    fn config(&self) -> Cow<'_, Self::Config> {
        Cow::Borrowed(&self.cfg)
    }
}
