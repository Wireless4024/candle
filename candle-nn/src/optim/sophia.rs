use crate::{tweaks::SafeTensorStorage, Optimizer, VarBuilder};
use candle::{
    backprop::GradStore,
    tweaks::{flags::disable_backprop, NamedVar},
    Tensor, Var,
};
// https://github.com/Liuhong99/Sophia/blob/main/sophia.py
// https://arxiv.org/pdf/2305.14342
#[derive(Debug, Clone, Copy)]
pub struct ParamSophia {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub rho: f64,
    pub weight_decay: f64,
    pub bs: f64,
    pub maximize: bool,
    pub capturable: bool,
    pub dynamic: bool,
}

impl Default for ParamSophia {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.965,
            beta2: 0.99,
            rho: 0.04,
            weight_decay: 1e-2,
            bs: 5210.0,
            maximize: false,
            capturable: false,
            dynamic: false,
        }
    }
}

struct VarSophia {
    var: NamedVar,
    exp_avg: Var,
    hessian: Var,
}

pub struct Sophia {
    vars: Vec<VarSophia>,
    params: ParamSophia,
    step_t: usize,
}

impl Sophia {
    fn update_hessian(&self, grad: &Tensor, var: &VarSophia) -> candle::Result<()> {
        let hessian = var.hessian.as_tensor();
        let hessian = (hessian * self.params.beta2)?;
        let hessian = hessian + (grad.sqr()? * (1.0 - self.params.beta2))?;
        var.hessian.set(&hessian?)?;
        Ok(())
    }

    fn update_exp_avg(&self, grad: &Tensor, var: &VarSophia) -> candle::Result<()> {
        let exp_avg = var.exp_avg.as_tensor();
        let exp_avg = (exp_avg * self.params.beta1)?;
        let alpha = 1.0 - self.params.beta1;
        let grad = (grad * alpha)?;
        let exp_avg = (exp_avg + grad)?;
        var.exp_avg.set(&exp_avg)?;
        Ok(())
    }
}

const EPSILON: f64 = 1e-15;

impl Optimizer for Sophia {
    type Config = ParamSophia;

    fn new(vars: Vec<Var>, params: Self::Config) -> candle::Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let exp_avg = Var::zeros(shape, dtype, &device)?;
                let hessian = Var::zeros(shape, dtype, &device)?;
                Ok(VarSophia {
                    var: var.into(),
                    exp_avg,
                    hessian,
                })
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
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
                let exp_avg = vb
                    .get_with_hints_dtype(shape, "exp_avg", crate::Init::Const(0.0), dtype)?
                    .to_device(device)?;
                let hessian = vb
                    .get_with_hints_dtype(shape, "hessian", crate::Init::Const(0.0), dtype)?
                    .to_device(device)?;
                Ok(VarSophia {
                    var,
                    exp_avg: Var::from_tensor(&exp_avg)?,
                    hessian: Var::from_tensor(&hessian)?,
                })
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn step(&mut self, grads: &GradStore) -> candle::Result<()> {
        let _backprop = disable_backprop();
        self.step_t += 1;
        for var in &self.vars {
            let theta = &var.var;
            if let Some(g) = grads.get(theta) {
                self.update_hessian(g, var)?;
                let grad = if self.params.maximize { g.neg()? } else { g.clone() };
                self.update_exp_avg(&grad, var)?;
                // some var like bias will request NoDecay which will skip this part
                let next_theta = if let Some(decay) = var.var.optimizer_hint().get_decay(self.params.weight_decay) {
                    let lr_lambda = self.params.lr * decay;
                    (theta.as_tensor() * (1f64 - lr_lambda))?
                } else {
                    theta.as_tensor().clone()
                };

                let step_size_neg = -self.params.lr;

                let divider = (var.hessian.as_tensor() * (self.params.rho * self.params.bs))?;
                let divider = (divider + EPSILON)?.clamp(f32::MIN, 1.0)?;
                let ratio = var.exp_avg.as_tensor().abs()? / divider;
                let next_theta = addcmul(&next_theta, &var.exp_avg.as_tensor().sign()?, &ratio?, step_size_neg)?;
                theta.set(&next_theta)?;
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

    fn load_state(&mut self, storage: &SafeTensorStorage) -> candle::Result<()> {
        if let Some(step_t) = storage.get_primitive("step_t")? {
            self.step_t = step_t;
        }
        Ok(())
    }

    fn save_state(&self, storage: &SafeTensorStorage) {
        storage.put_primitive("step_t", self.step_t);
    }
}

fn addcmul(base: &Tensor, t1: &Tensor, t2: &Tensor, value: f64) -> candle::Result<Tensor> {
    let buf = (value * t1)?;
    let buf = (buf * t2)?;
    base + buf
}
