use candle::tweaks::NamedVar;

mod serializable;
pub mod storage;

#[allow(unused_imports)]
pub use serializable::{get_config, LoadParam, ModuleRegistry, SerializableModule, SerializableSequential};
pub use storage::*;

impl crate::VarMap {
    /// Retrieve all the variables currently stored in the map.
    pub fn all_vars_named(&self) -> Vec<NamedVar> {
        let tensor_data = self.data().lock().unwrap();
        #[allow(clippy::map_clone)]
        tensor_data
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()).into())
            .collect::<Vec<_>>()
    }
}

pub fn register_builtin_serializable_modules(registry: &mut ModuleRegistry) {
    use crate::{rnn::*, Activation, BatchNorm, Dropout, Linear};
    registry.register::<BatchNorm>();
    registry.register::<Linear>();
    registry.register::<Activation>();
    registry.register::<Dropout>();
    registry.register::<SerializableSequential>();
    registry.register::<GRUStack>();
    registry.register::<GRU>();
    registry.register::<LSTM>();
}
