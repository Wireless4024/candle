//! Recurrent Neural Networks

use std::borrow::Cow;
use crate::{linear, linear_no_bias, Dropout, Linear, VarBuilder};
use candle::{tweaks::ParameterCount, DType, Device, Error, IndexOp, Module, ModuleT, Result, Tensor};
use candle::tweaks::flags::set_training;
use crate::tweaks::{ModuleRegistry, SerializableModule};

/// Trait for Recurrent Neural Networks.
#[allow(clippy::upper_case_acronyms)]
pub trait RNN {
    type State: Clone;

    /// A zero state from which the recurrent network is usually initialized.
    fn zero_state(&self, batch_dim: usize) -> Result<Self::State>;

    /// Applies a single step of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, features].
    fn step(&self, input: &Tensor, state: &Self::State) -> Result<Self::State>;

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    /// The initial state is the result of applying zero_state.
    fn seq(&self, input: &Tensor) -> Result<Vec<Self::State>> {
        let batch_dim = input.dim(0)?;
        let state = self.zero_state(batch_dim)?;
        self.seq_init(input, &state)
    }

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    fn seq_init(&self, input: &Tensor, init_state: &Self::State) -> Result<Vec<Self::State>> {
        let (_b_size, seq_len, _features) = input.dims3()?;
        let mut output = Vec::with_capacity(seq_len);
        for seq_index in 0..seq_len {
            let input = input.i((.., seq_index, ..))?.contiguous()?;
            let state = if seq_index == 0 {
                self.step(&input, init_state)?
            } else {
                self.step(&input, &output[seq_index - 1])?
            };
            output.push(state);
        }
        Ok(output)
    }

    /// Converts a sequence of state to a tensor.
    fn states_to_tensor<'a, I: IntoIterator<Item = &'a Self::State>>(&self, states: I) -> Result<Tensor>
    where
        I::IntoIter: ExactSizeIterator,
        Self::State: 'a;
}

pub struct RNNModule<R: RNN> {
    rnn: R,
}

impl<R: RNN> RNNModule<R> {
    pub fn new(rnn: R) -> Self {
        Self { rnn }
    }
}

impl<R: RNN> Module for RNNModule<R> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.rank() == 2 {
            let batch = xs.dim(0)?;
            let state = self.rnn.zero_state(batch)?;
            let state = self.rnn.step(xs, &state)?;
            self.rnn.states_to_tensor(Some(&state))
        } else {
            let state = self.rnn.seq(xs)?;
            self.rnn.states_to_tensor(state.last())
        }
    }
}

/// The state for a LSTM network, this contains two tensors.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct LSTMState {
    pub h: Tensor,
    pub c: Tensor,
}

impl LSTMState {
    pub fn new(h: Tensor, c: Tensor) -> Self {
        LSTMState { h, c }
    }

    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> &Tensor {
        &self.h
    }

    /// The cell state vector.
    pub fn c(&self) -> &Tensor {
        &self.c
    }
}

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
pub enum Direction {
    Forward,
    Backward,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
pub struct LSTMConfig {
    pub w_ih_init: super::Init,
    pub w_hh_init: super::Init,
    pub b_ih_init: Option<super::Init>,
    pub b_hh_init: Option<super::Init>,
    pub layer_idx: usize,
    pub direction: Direction,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(super::Init::Const(0.)),
            b_hh_init: Some(super::Init::Const(0.)),
            layer_idx: 0,
            direction: Direction::Forward,
        }
    }
}

impl LSTMConfig {
    pub fn default_no_bias() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: None,
            b_hh_init: None,
            layer_idx: 0,
            direction: Direction::Forward,
        }
    }
}

/// A Long Short-Term Memory (LSTM) layer.
///
/// <https://en.wikipedia.org/wiki/Long_short-term_memory>
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct LSTM {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
    hidden_dim: usize,
    layer_idx: usize,
    config: LSTMConfig,
    device: Device,
    dtype: DType,
}

impl LSTM {
    /// Creates a LSTM layer.
    pub fn new(in_dim: usize, hidden_dim: usize, config: LSTMConfig, vb: crate::VarBuilder) -> Result<Self> {
        let layer_idx = config.layer_idx;
        let direction_str = match config.direction {
            Direction::Forward => "",
            Direction::Backward => "_reverse",
        };
        let w_ih = {
            let _hint = candle::tweaks::with_var_kind(candle::tweaks::VariableKind::GeometryAware);
            vb.get_with_hints(
                (4 * hidden_dim, in_dim),
                &format!("weight_ih_l{layer_idx}{direction_str}"), // Only a single layer is supported.
                config.w_ih_init,
            )?
        };
        let w_hh = {
            let _hint = candle::tweaks::with_var_kind(candle::tweaks::VariableKind::GeometryAware);
            vb.get_with_hints(
                (4 * hidden_dim, hidden_dim),
                &format!("weight_hh_l{layer_idx}{direction_str}"), // Only a single layer is supported.
                config.w_hh_init,
            )?
        };
        let b_ih = match config.b_ih_init {
            Some(init) => {
                let _hint = candle::tweaks::with_var_kind(candle::tweaks::VariableKind::NoDecay);
                Some(vb.get_with_hints(4 * hidden_dim, &format!("bias_ih_l{layer_idx}{direction_str}"), init)?)
            }
            None => None,
        };
        let b_hh = match config.b_hh_init {
            Some(init) => {
                let _hint = candle::tweaks::with_var_kind(candle::tweaks::VariableKind::NoDecay);
                Some(vb.get_with_hints(4 * hidden_dim, &format!("bias_hh_l{layer_idx}{direction_str}"), init)?)
            }
            None => None,
        };
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_dim,
            layer_idx,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn config(&self) -> &LSTMConfig {
        &self.config
    }
}

/// Creates a LSTM layer.
pub fn lstm(in_dim: usize, hidden_dim: usize, config: LSTMConfig, vb: crate::VarBuilder) -> Result<LSTM> {
    LSTM::new(in_dim, hidden_dim, config, vb)
}

impl RNN for LSTM {
    type State = LSTMState;

    fn zero_state(&self, batch_dim: usize) -> Result<Self::State> {
        let zeros = Tensor::zeros((batch_dim, self.hidden_dim), self.dtype, &self.device)?.contiguous()?;
        Ok(Self::State {
            h: zeros.clone(),
            c: zeros.clone(),
        })
    }

    fn step(&self, input: &Tensor, in_state: &Self::State) -> Result<Self::State> {
        let w_ih = input.matmul(&self.w_ih.t()?)?;
        let w_hh = in_state.h.matmul(&self.w_hh.t()?)?;
        let w_ih = match &self.b_ih {
            None => w_ih,
            Some(b_ih) => w_ih.broadcast_add(b_ih)?,
        };
        let w_hh = match &self.b_hh {
            None => w_hh,
            Some(b_hh) => w_hh.broadcast_add(b_hh)?,
        };
        let chunks = (&w_ih + &w_hh)?.chunk(4, 1)?;
        let in_gate = crate::ops::sigmoid(&chunks[0])?;
        let forget_gate = crate::ops::sigmoid(&chunks[1])?;
        let cell_gate = chunks[2].tanh()?;
        let out_gate = crate::ops::sigmoid(&chunks[3])?;

        let next_c = ((forget_gate * &in_state.c)? + (in_gate * cell_gate)?)?;
        let next_h = (out_gate * next_c.tanh()?)?;
        Ok(LSTMState { c: next_c, h: next_h })
    }

    fn states_to_tensor<'a, I: IntoIterator<Item = &'a Self::State>>(&self, states: I) -> Result<Tensor>
    where
        I::IntoIter: ExactSizeIterator,
        Self::State: 'a,
    {
        let mut states = states.into_iter();
        if states.len() == 1 {
            return Ok(states.next().unwrap().h.clone());
        }
        let states = states.map(|s| s.h.clone()).collect::<Vec<_>>();
        Tensor::stack(&states, 1)
    }
}

// -
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy)]
pub struct LSTMInitializeConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub layer_idx: usize,
    #[serde(flatten)]
    pub config: LSTMConfig,
}

impl ModuleT for LSTM {
    fn forward_t(&self, xs: &Tensor, _: bool) -> Result<Tensor> {
        let batch_size = xs.dim(2)?;
        let state = self.zero_state(batch_size)?;
        Ok(self.step(xs, &state)?.h)
    }
}

impl SerializableModule for LSTM {
    type Config = LSTMInitializeConfig;

    fn load(config: Self::Config, _: &ModuleRegistry, vb: VarBuilder) -> std::result::Result<Self, Error> {
        lstm(config.input_dim, config.hidden_dim, config.config, vb)
    }

    fn config(&self) -> Cow<'_, Self::Config> {
        Cow::Owned(LSTMInitializeConfig {
            input_dim: self.w_ih.get_linear_shape().unwrap().0,
            hidden_dim: self.hidden_dim,
            layer_idx: self.layer_idx,
            config: self.config,
        })
    }
}

impl ParameterCount for LSTM {
    fn parameter_count(&self) -> usize {
        self.w_ih.parameter_count()
            + self.w_hh.parameter_count()
            + self.b_ih.as_ref().map_or(0, |b| b.parameter_count())
            + self.b_hh.as_ref().map_or(0, |b| b.parameter_count())
    }
}

/// The state for a GRU network, this contains a single tensor.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct GRUState {
    pub h: Tensor,
}

impl GRUState {
    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> &Tensor {
        &self.h
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
pub struct GRUConfig {
    pub w_ih_init: super::Init,
    pub w_hh_init: super::Init,
    pub b_ih_init: Option<super::Init>,
    pub b_hh_init: Option<super::Init>,
}

impl Default for GRUConfig {
    fn default() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(super::Init::Const(0.)),
            b_hh_init: Some(super::Init::Const(0.)),
        }
    }
}

impl GRUConfig {
    pub fn default_no_bias() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: None,
            b_hh_init: None,
        }
    }
}

/// A Gated Recurrent Unit (GRU) layer.
///
/// <https://en.wikipedia.org/wiki/Gated_recurrent_unit>
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct GRU {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
    hidden_dim: usize,
    config: GRUConfig,
    device: Device,
    dtype: DType,
}

impl GRU {
    /// Creates a GRU layer.
    pub fn new(in_dim: usize, hidden_dim: usize, config: GRUConfig, vb: crate::VarBuilder) -> Result<Self> {
        let w_ih = {
            let _hint = candle::tweaks::with_var_kind(candle::tweaks::VariableKind::GeometryAware);
            vb.get_with_hints(
                (3 * hidden_dim, in_dim),
                "weight_ih_l0", // Only a single layer is supported.
                config.w_ih_init,
            )?
        };
        let w_hh = {
            let _hint = candle::tweaks::with_var_kind(candle::tweaks::VariableKind::GeometryAware);
            vb.get_with_hints(
                (3 * hidden_dim, hidden_dim),
                "weight_hh_l0", // Only a single layer is supported.
                config.w_hh_init,
            )?
        };
        let b_ih = match config.b_ih_init {
            Some(init) => {
                let _hint = candle::tweaks::with_var_kind(candle::tweaks::VariableKind::NoDecay);
                Some(vb.get_with_hints(3 * hidden_dim, "bias_ih_l0", init)?)
            }
            None => None,
        };
        let b_hh = match config.b_hh_init {
            Some(init) => {
                let _hint = candle::tweaks::with_var_kind(candle::tweaks::VariableKind::NoDecay);
                Some(vb.get_with_hints(3 * hidden_dim, "bias_hh_l0", init)?)
            }
            None => None,
        };
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_dim,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn config(&self) -> &GRUConfig {
        &self.config
    }
}

pub fn gru(in_dim: usize, hidden_dim: usize, config: GRUConfig, vb: crate::VarBuilder) -> Result<GRU> {
    GRU::new(in_dim, hidden_dim, config, vb)
}

impl RNN for GRU {
    type State = GRUState;

    fn zero_state(&self, batch_dim: usize) -> Result<Self::State> {
        let h = Tensor::zeros((batch_dim, self.hidden_dim), self.dtype, &self.device)?.contiguous()?;
        Ok(Self::State { h })
    }

    fn step(&self, input: &Tensor, in_state: &Self::State) -> Result<Self::State> {
        let w_ih = input.matmul(&self.w_ih.t()?)?;
        let w_hh = in_state.h.matmul(&self.w_hh.t()?)?;
        let w_ih = match &self.b_ih {
            None => w_ih,
            Some(b_ih) => w_ih.broadcast_add(b_ih)?,
        };
        let w_hh = match &self.b_hh {
            None => w_hh,
            Some(b_hh) => w_hh.broadcast_add(b_hh)?,
        };
        let chunks_ih = w_ih.chunk(3, 1)?;
        let chunks_hh = w_hh.chunk(3, 1)?;
        let r_gate = crate::ops::sigmoid(&(&chunks_ih[0] + &chunks_hh[0])?)?;
        let z_gate = crate::ops::sigmoid(&(&chunks_ih[1] + &chunks_hh[1])?)?;
        let n_gate = (&chunks_ih[2] + (r_gate * &chunks_hh[2])?)?.tanh();

        let next_h = ((&z_gate * &in_state.h)? - ((&z_gate - 1.)? * n_gate)?)?;
        Ok(GRUState { h: next_h })
    }

    fn states_to_tensor<'a, I: IntoIterator<Item = &'a Self::State>>(&self, states: I) -> Result<Tensor>
    where
        I::IntoIter: ExactSizeIterator,
        Self::State: 'a,
    {
        let mut states = states.into_iter();
        if states.len() == 1 {
            return Ok(states.next().unwrap().h.clone());
        }
        let states = states.map(|s| s.h.clone()).collect::<Vec<_>>();
        Tensor::cat(&states, 1)
    }
}

// -
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy)]
pub struct GRUInitializeConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    #[serde(flatten)]
    pub config: GRUConfig,
}

impl ModuleT for GRU {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let batch_size = xs.dim(2)?;
        let state = self.zero_state(batch_size)?;
        Ok(self.step(xs, &state)?.h)
    }
}

impl SerializableModule for GRU {
    type Config = GRUInitializeConfig;

    fn load(config: Self::Config, _: &ModuleRegistry, vb: VarBuilder) -> std::result::Result<Self, Error> {
        GRU::new(config.input_dim, config.hidden_dim, config.config, vb)
    }

    fn config(&self) -> Cow<'_, Self::Config> {
        let (input_dim, hidden_dim) = self.w_ih.get_linear_shape().unwrap();
        Cow::Owned(GRUInitializeConfig {
            input_dim,
            hidden_dim,
            config: self.config,
        })
    }
}

impl ParameterCount for GRU {
    fn parameter_count(&self) -> usize {
        self.w_hh.parameter_count()
            + self.w_ih.parameter_count()
            + self.b_hh.as_ref().map_or(0, |b| b.parameter_count())
            + self.b_ih.as_ref().map_or(0, |b| b.parameter_count())
    }
}

pub struct GRUStack {
    projection: Linear,
    layers: Vec<GRU>,
    head: Linear,
    dropout: Dropout,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy)]
pub struct GRUStackConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub n_layers: usize,
    pub gru: GRUConfig,
    pub dropout: f32,
}

impl GRUStack {
    pub fn new(cfg: GRUStackConfig, vb: crate::VarBuilder) -> Result<Self> {
        let projection = linear_no_bias(cfg.input_dim, cfg.hidden_dim, vb.pp("projection"))?;
        let rnn = (0..cfg.n_layers)
            .map(|idx| gru(cfg.hidden_dim, cfg.hidden_dim, cfg.gru, vb.pp(idx)))
            .collect::<Result<Vec<_>>>()?;
        if cfg.n_layers == 0 {
            candle::bail!("At least one layer is required.");
        }
        let head = linear(cfg.hidden_dim, cfg.output_dim, vb.pp("head"))?;
        let dropout = Dropout::new(cfg.dropout);
        Ok(Self {
            projection,
            layers: rnn,
            head,
            dropout,
        })
    }

    fn step(&self, input: &Tensor, state: &GRUState, train: bool) -> Result<GRUState> {
        let mut state = self.layers[0].step(input, state)?;
        let mut idx = 1;
        while idx < self.layers.len() {
            state = self.layers[idx].step(&state.h, &state)?;
            state = GRUState {
                h: self.dropout.forward_t(&state.h, train)?,
            };
            idx += 1;
        }
        Ok(state)
    }
}

impl Module for GRUStack {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_(xs)
    }
}

impl ModuleT for GRUStack {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let _training = set_training(train);
        let (b_size, seq_len, _) = xs.dims3()?;
        let xs = self.projection.forward(xs)?;
        let mut state = self.layers[0].zero_state(b_size)?;
        for seq_index in 0..seq_len {
            let xs = xs.i((.., seq_index, ..))?.contiguous()?;
            state = self.step(&xs, &state, train)?;
        }
        self.head.forward(&state.h)
    }
}

impl SerializableModule for GRUStack {
    type Config = GRUStackConfig;

    fn load(config: Self::Config, _: &ModuleRegistry, vb: VarBuilder) -> std::result::Result<Self, Error> {
        Self::new(config, vb)
    }

    fn config(&self) -> Cow<'_, Self::Config> {
        Cow::Owned(GRUStackConfig{
            input_dim: self.projection.config().in_features,
            hidden_dim: self.projection.config().out_features,
            output_dim: self.head.config().out_features,
            n_layers: self.layers.len(),
            gru: self.layers[0].config,
            dropout: self.dropout.config().into_owned(),
        })
    }
}

impl ParameterCount for GRUStack {
    fn parameter_count(&self) -> usize {
        self.projection.parameter_count() + self.head.parameter_count() + self.layers.parameter_count()
    }
}
