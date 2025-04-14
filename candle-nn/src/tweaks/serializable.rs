mod seq;

pub use seq::SerializableSequential;

use ahash::HashMap;
use candle::{tweaks::module::DynModule, ModuleT};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::{any::type_name, borrow::Cow, fmt::Debug};

pub struct ModuleRegistry {
    registry: HashMap<&'static str, FnLoadModule>,
}

impl ModuleRegistry {
    pub fn new() -> Self {
        use crate::{Activation, BatchNorm, Dropout, Linear};
        let mut registry = HashMap::<&'static str, FnLoadModule>::default();
        registry.insert(BatchNorm::name(), BatchNorm::load_boxed);
        registry.insert(Linear::name(), Linear::load_boxed);
        registry.insert(Activation::name(), Activation::load_boxed);
        registry.insert(Dropout::name(), Dropout::load_boxed);
        registry.insert(SerializableSequential::name(), SerializableSequential::load_boxed);
        Self { registry }
    }

    pub fn register<M: SerializableModule>(&mut self) {
        self.registry.insert(M::name(), M::load_boxed);
    }

    pub fn load(&self, param: LoadParam, vb: &crate::VarBuilder) -> Result<DynModule, candle::Error> {
        let vb = vb.pp(&param.name);
        let load_fn = self
            .registry
            .get(&*param.module)
            .ok_or_else(|| candle::Error::msg(format!("Unregistered module: {}", &param.module)))?;
        load_fn(param.param, self, vb)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LoadParam {
    pub name: String,
    pub module: String,
    pub param: Value,
}

#[allow(dead_code)]
pub trait SerializableModule: ModuleT + Sized + 'static {
    type Config: Serialize + DeserializeOwned + Clone + Debug;
    fn name() -> &'static str {
        type_name::<Self>().rsplit("::").next().unwrap()
    }
    fn load_boxed(config: Value, registry: &ModuleRegistry, vb: crate::VarBuilder) -> Result<DynModule, candle::Error> {
        Ok(DynModule::new(Self::load(
            serde_json::from_value(config).map_err(candle::Error::wrap)?,
            registry,
            vb,
        )?))
    }
    fn load(config: Self::Config, registry: &ModuleRegistry, vb: crate::VarBuilder) -> Result<Self, candle::Error>;
    fn config(&self) -> Cow<'_, Self::Config>;
    fn config_value(&self, name: impl ToString) -> Value {
        serde_json::json! {{
            "module": Self::name(),
            "name": name.to_string(),
            "param": serde_json::to_value(self.config()).unwrap(),
        }}
    }
    fn is_serializable() -> bool {
        true
    }
}

pub fn get_config<M: 'static>(module: &M) -> Option<Value> {
    trait GetConfigInner {
        fn get_config(&self) -> Option<Value>;
    }

    impl<M: 'static> GetConfigInner for M {
        default fn get_config(&self) -> Option<Value> {
            None
        }
    }

    impl<M: SerializableModule> GetConfigInner for M {
        fn get_config(&self) -> Option<Value> {
            if M::is_serializable() {
                Some(serde_json::to_value(self.config()).ok()?)
            } else {
                None
            }
        }
    }
    module.get_config()
}

#[allow(dead_code)]
pub type FnLoadModule = fn(Value, &ModuleRegistry, crate::VarBuilder) -> Result<DynModule, candle::Error>;
