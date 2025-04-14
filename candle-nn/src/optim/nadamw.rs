use crate::{
    optim::{gradclip::GradientClipping, OptimizerExt},
    Optimizer, VarBuilder,
};
use ahash::HashSet;
use candle::{
    backprop::GradStore,
    tweaks::{ArcStr, NamedVar},
    Tensor, Var,
};
use candle::tweaks::flags::disable_backprop;

#[derive(Copy, Clone, Debug)]
pub struct ParamsNAdamW {
    pub lr: f64,
    pub beta1: f64,
    /// lowering β₂ from 0.999 → 0.98 makes the second-moment (variance) estimate react much faster to changes in your
    /// gradient statistics. That’s usually helpful on non-stationary, highly overlapping sequence windows.
    ///
    /// β₂=0.98 ⇒ effective window ≈ 1/(1−β₂) ≈ 50 steps; half-life ≈ 34 steps.
    ///
    /// β₂=0.999 ⇒ effective window ≈ 1000 steps; half-life ≈ 693 steps.
    /// So with 0.999, your RMS denominator changes very slowly and can under-react to regime shifts;
    /// with 0.98 it adapts quickly to overlapping data.
    ///
    /// # When 0.999 can be better
    ///
    /// Massive batches with fairly stationary gradients (e.g., image classification on shuffled IID data) where you
    /// want a very smooth variance estimate.
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub grad_clip: GradientClipping,
    pub nesterov: bool,
}

impl ParamsNAdamW {
    pub fn effective_window(&self) -> f64 {
        1.0 / (1.0 - self.beta2)
    }

    pub fn half_life(&self) -> f64 {
        (0.5f64.ln()) / self.beta2.ln()
    }
}

impl Default for ParamsNAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.98,
            eps: 1e-8,
            weight_decay: 0.001,
            grad_clip: GradientClipping::None,
            nesterov: true,
        }
    }
}

#[derive(Debug)]
pub struct VarNAdamW {
    var: NamedVar,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct NAdamW {
    vars: Vec<VarNAdamW>,
    step_t: usize,
    params: ParamsNAdamW,
    missing_grad: HashSet<ArcStr>,
}

#[derive(Debug, Clone, Copy)]
pub struct NAdamWStepOpt {
    pub lr: f64,
    pub scale_m: f64,
    pub scale_v: f64,
}

impl Optimizer for NAdamW {
    type Config = ParamsNAdamW;

    fn new(vars: Vec<Var>, params: ParamsNAdamW) -> candle::Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarNAdamW {
                    var: var.into(),
                    first_moment,
                    second_moment,
                })
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
            missing_grad: Default::default(),
        })
    }

    fn new2(vars: Vec<NamedVar>, params: Self::Config, vb: VarBuilder) -> candle::Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let vb = vb.pp(var.name());
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = vb
                    .get_with_hints_dtype(shape, "first_moment", crate::Init::Const(0.0), dtype)?
                    .to_device(device)?;
                let second_moment = vb
                    .get_with_hints_dtype(shape, "second_moment", crate::Init::Const(0.0), dtype)?
                    .to_device(device)?;
                Ok(VarNAdamW {
                    var,
                    first_moment: Var::from_tensor(&first_moment)?,
                    second_moment: Var::from_tensor(&second_moment)?,
                })
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
            missing_grad: Default::default(),
        })
    }

    fn step(&mut self, grads: &GradStore) -> candle::Result<()> {
        let _backprop = disable_backprop();
        self.step_t += 1;
        let lr = self.params.lr;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        let opt = NAdamWStepOpt { lr, scale_m, scale_v };
        for var in &self.vars {
            let theta = &var.var;
            if let Some(g) = grads.get(theta) {
                self.step_once(g, var, &opt)?;
            } else if !self.missing_grad.contains(theta.name()) {
                println!("No grad for {}", theta.name());
                self.missing_grad.insert(theta.name().clone());
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn zeroed(&mut self) -> candle::Result<()> {
        for var in &self.vars {
            var.first_moment.zero_set()?;
            var.second_moment.zero_set()?;
        }
        Ok(())
    }

    fn load_state(&mut self, storage: &crate::tweaks::SafeTensorStorage) -> candle::Result<()> {
        if let Some(step_t) = storage.get_primitive("step_t")? {
            self.step_t = step_t;
        }
        if let Some(lr) = storage.get_primitive("lr")? {
            self.set_learning_rate(lr);
        }
        Ok(())
    }

    fn save_state(&self, storage: &crate::tweaks::SafeTensorStorage) {
        storage.put_primitive("step_t", self.step_t);
        storage.put_primitive("lr", self.learning_rate());
    }
}

impl OptimizerExt for NAdamW {
    type StepOpt = NAdamWStepOpt;
    type Var = VarNAdamW;

    fn vars(&self) -> &[Self::Var] {
        &self.vars
    }

    fn step_opt(&self) -> Self::StepOpt {
        NAdamWStepOpt {
            lr: self.params.lr,
            scale_m: 1f64 / (1f64 - self.params.beta1.powi(self.step_t as i32)),
            scale_v: 1f64 / (1f64 - self.params.beta2.powi(self.step_t as i32)),
        }
    }

    fn step_once(&self, gradient: &Tensor, var: &Self::Var, opt: &Self::StepOpt) -> candle::Result<()> {
        let gradient = self.params.grad_clip.forward(gradient, var.var.name())?;
        let theta = &var.var;
        let m = &var.first_moment;
        let v = &var.second_moment;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let &NAdamWStepOpt { lr, scale_m, scale_v } = opt;

        let next_m = ((m.as_tensor() * beta1)? + (&gradient * (1.0 - beta1))?)?;
        let next_v = ((v.as_tensor() * beta2)? + (gradient.sqr()? * (1.0 - beta2))?)?;
        let m_hat = (&next_m * scale_m)?;
        let v_hat = (&next_v * scale_v)?;

        // some var like bias will request NoDecay which will skip this part
        let next_theta = if let Some(decay) = var.var.optimizer_hint().get_decay(self.params.weight_decay) {
            let lr_lambda = lr * decay;
            (theta.as_tensor() * (1f64 - lr_lambda))?
        } else {
            theta.as_tensor().clone()
        };

        // Denominator (shared by Adam and NAdam)
        let denom = (v_hat.sqrt()? + self.params.eps)?;

        // Numerator: Adam vs NAdam
        let numer = if self.params.nesterov {
            // NAdam numerator: β1·m̂ + ((1−β1)/(1−β1^t))·g_t
            // Note: scale_m == 1/(1−β1^t)
            ((&m_hat * beta1)? + (&gradient * ((1.0 - beta1) * scale_m))?)?
        } else {
            // Standard Adam numerator: m̂
            m_hat
        };

        let adjusted_grad = (&numer / &denom)?;
        let next_theta = (next_theta - (adjusted_grad * lr)?)?;
        m.set(&next_m)?;
        v.set(&next_v)?;
        theta.set(&next_theta)?;
        Ok(())
    }
}

impl NAdamW {
    pub fn new_lr(vars: Vec<NamedVar>, learning_rate: f64, vb: VarBuilder) -> candle::Result<Self> {
        let params = ParamsNAdamW {
            lr: learning_rate,
            ..ParamsNAdamW::default()
        };
        Self::new2(vars, params, vb)
    }

    pub fn params(&self) -> &ParamsNAdamW {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsNAdamW) {
        self.params = params;
    }

    pub fn get_step_t(&self) -> usize {
        self.step_t
    }

    pub fn set_step_t(&mut self, step_t: usize) {
        self.step_t = step_t;
    }
}
