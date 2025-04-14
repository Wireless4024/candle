use crate::VarMap;
use candle::tweaks::NamedVar;

mod serializable;
pub mod storage;

#[allow(unused_imports)]
pub use serializable::{get_config, LoadParam, ModuleRegistry, SerializableModule, SerializableSequential};
pub use storage::*;

impl VarMap {
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
