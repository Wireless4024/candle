//! Embedding Layer.

use std::borrow::Cow;
use candle::{Result, Tensor};

#[derive(Clone, Debug)]
pub struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    /// Get the hidden size of the embedding matrix
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl crate::Module for Embedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

pub fn embedding(in_size: usize, out_size: usize, vb: crate::VarBuilder) -> Result<Embedding> {
    let _kind = candle::tweaks::with_var_kind(candle::tweaks::VariableKind::with_decay_multiplier(0.02));
    let embeddings = vb.get_with_hints(
        (in_size, out_size),
        "weight",
        crate::Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    )?;
    Ok(Embedding::new(embeddings, out_size))
}

// --
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct EmbeddingConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl crate::tweaks::SerializableModule for Embedding {
    type Config = EmbeddingConfig;

    fn load(config: Self::Config, _: &crate::tweaks::ModuleRegistry, vb: crate::VarBuilder) -> std::result::Result<Self, candle::Error> {
        embedding(config.vocab_size, config.hidden_size, vb)
    }

    fn config(&self) -> Cow<'_, Self::Config> {
        Cow::Owned(EmbeddingConfig {
            vocab_size: self.embeddings.shape().dims()[0],
            hidden_size: self.hidden_size,
        })
    }
}

impl candle::tweaks::ParameterCount for Embedding {
    fn parameter_count(&self) -> usize {
        self.embeddings.parameter_count()
    }
}