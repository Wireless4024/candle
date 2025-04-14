use crate::VarMap;
use candle::tweaks::NamedVar;

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
