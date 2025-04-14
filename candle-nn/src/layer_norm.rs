//! Layer Normalization.
//!
//! This layer applies Layer Normalization over a mini-batch of inputs as described in [`Layer
//! Normalization`]. The input is expected to have three dimensions: a batch dimension, a length,
//! and a hidden size, the normalization is applied over the last dimension.
//!
//! # Example
//!
//! ```rust
//! use candle::{Tensor, Device::Cpu, test_utils::to_vec3_round};
//! use candle_nn::{LayerNorm, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[1f32, 1f32, 1f32], &Cpu)?;
//! let b = Tensor::new(&[0f32, 0f32, 0f32], &Cpu)?;
//! let layer = LayerNorm::new(w, b, 1e-5);
//!
//! let xs = Tensor::new(
//!     &[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]],
//!     &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(
//!     to_vec3_round(&ys, 4)?,
//!     &[[[-1.2247, 0.0,  1.2247],
//!        [-1.2247, 0.0,  1.2247],
//!        [ 1.2247, 0.0, -1.2247]]]);
//! # Ok(()) }
//! ```
//!
//! [`Layer Normalization`]: https://arxiv.org/abs/1607.06450

use std::borrow::Cow;
use candle::{DType, Error, Module, Result, Tensor, D};
use crate::VarBuilder;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LayerNormConfig {
    #[serde(default = "default_eps")]
    pub eps: f64,
    /// Whether to remove the mean or not, the default is true and when set to false, this turns
    /// this layer into RmsNorm.
    #[serde(default = "default_remove_mean")]
    pub remove_mean: bool,
    #[serde(default = "default_affine")]
    pub affine: bool,
}

fn default_eps() -> f64 {
    1e-5
}

fn default_remove_mean() -> bool {
    true
}

fn default_affine() -> bool {
    true
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        }
    }
}

impl From<f64> for LayerNormConfig {
    fn from(eps: f64) -> Self {
        Self {
            eps,
            remove_mean: true,
            affine: true,
        }
    }
}

// This layer norm version handles both weight and bias so removes the mean.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    remove_mean: bool,
    eps: f64,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: Some(bias),
            remove_mean: true,
            eps,
        }
    }

    pub fn new_no_bias(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            remove_mean: true,
            eps,
        }
    }

    pub fn rms_norm(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            remove_mean: false,
            eps,
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.is_contiguous() && self.remove_mean {
            if let Some(bias) = self.bias.as_ref() {
                return crate::ops::layer_norm(x, &self.weight, bias, self.eps as f32);
            }
        }
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let x = if self.remove_mean {
            let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn layer_norm<C: Into<LayerNormConfig>>(
    size: usize,
    config: C,
    vb: crate::VarBuilder,
) -> Result<LayerNorm> {
    let config = config.into();
    let weight = {
        let _kind = candle::tweaks::with_var_kind(candle::tweaks::VariableKind::NoDecay);
        vb.get_with_hints(size, "weight", crate::Init::Const(1.))?
    };
    let bias = if config.affine {
        Some(vb.get_with_hints(size, "bias", crate::Init::Const(0.))?)
    } else {
        None
    };
    Ok(LayerNorm {
        weight,
        bias,
        remove_mean: config.remove_mean,
        eps: config.eps,
    })
}

pub fn layer_norm_no_bias(size: usize, eps: f64, vb: crate::VarBuilder) -> Result<LayerNorm> {
    let config = LayerNormConfig {
        eps,
        remove_mean: true,
        affine: false,
    };
    layer_norm(size, config, vb)
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct LayerNormConfigWrapper {
    size: usize,
    #[serde(flatten)]
    inner: LayerNormConfig,
}

impl crate::tweaks::SerializableModule for LayerNorm {
    type Config = LayerNormConfigWrapper;

    fn load(config: Self::Config, _: &crate::tweaks::ModuleRegistry, vb: VarBuilder) -> std::result::Result<Self, Error> {
        layer_norm(config.size, config.inner, vb)
    }

    fn config(&self) -> Cow<'_, Self::Config> {
        Cow::Owned(LayerNormConfigWrapper {
            size: self.weight.dims()[0],
            inner: LayerNormConfig{
                eps: self.eps,
                remove_mean: self.remove_mean,
                affine: self.bias.is_some(),
            },
        })
    }
}

impl candle::tweaks::ParameterCount for LayerNorm {
    fn parameter_count(&self) -> usize {
        self.weight.parameter_count()
    }
}

/// RmsNorm is a specialized version of the LayerNorm module.
#[derive(Clone, Debug)]
pub struct RmsNorm(LayerNorm);

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self(LayerNorm::rms_norm(weight, eps))
    }

    pub fn into_inner(self) -> LayerNorm {
        self.0
    }

    /// Faster variant of the forward kernel, this can only be used on contiguous tensors though.
    pub fn forward_diff(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.is_contiguous() {
            crate::ops::rms_norm(xs, &self.0.weight, self.0.eps as f32)
        } else {
            self.0.forward(xs)
        }
    }
}

pub fn rms_norm(size: usize, eps: f64, vb: crate::VarBuilder) -> Result<RmsNorm> {
    let config = LayerNormConfig {
        eps,
        remove_mean: false,
        affine: false,
    };
    Ok(RmsNorm(layer_norm(size, config, vb)?))
}

// --
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct RmsNormConfig {
    pub size: usize,
    pub eps: f64,
}

impl crate::tweaks::SerializableModule for RmsNorm {
    type Config = RmsNormConfig;

    fn load(config: Self::Config, _: &crate::tweaks::ModuleRegistry, vb: VarBuilder) -> std::result::Result<Self, candle::Error> {
        rms_norm(config.size, config.eps, vb)
    }

    fn config(&self) -> Cow<'_, Self::Config> {
        Cow::Owned(RmsNormConfig {
            size: self.0.weight.dims()[0],
            eps: self.0.eps,
        })
    }
}

impl candle::tweaks::ParameterCount for RmsNorm {
    fn parameter_count(&self) -> usize {
        self.0.weight.parameter_count()
    }
}