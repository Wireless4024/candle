use crate::{Optimizer, VarBuilder};
use candle::{
    backprop::GradStore,
    tweaks::{flags::disable_backprop, NamedVar},
    Tensor, Var,
};
// https://github.com/Liuhong99/Sophia/blob/main/sophia.py
// https://arxiv.org/pdf/2305.14342
#[derive(Debug, Clone, Copy)]
pub struct ParamLion {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
}

impl Default for ParamLion {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 1e-2,
        }
    }
}

struct VarLion {
    var: NamedVar,
    exp_avg: Var,
}

pub struct Lion {
    vars: Vec<VarLion>,
    params: ParamLion,
}

// https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/lion_pytorch.py
fn lion_update(var: &VarLion, grad: &Tensor, params: &ParamLion) -> candle::Result<()> {
    let theta = &var.var;
    // some var like bias will request NoDecay which will skip this part
    let next_theta = if let Some(decay) = var.var.optimizer_hint().get_decay(params.weight_decay) {
        let lr_lambda = params.lr * decay;
        (theta.as_tensor() * (1f64 - lr_lambda))?
    } else {
        theta.as_tensor().clone()
    };
    let exp_avg = (var.exp_avg.as_tensor() * params.beta1)?;
    let grad_beta1 = (grad * (1.0 - params.beta1))?;
    let update_sign = (exp_avg + grad_beta1)?.sign()?;
    let update = (update_sign * params.lr)?;
    let next_theta = (next_theta + update)?;
    theta.set(&next_theta)?;

    let grad_beta2 = (grad * (1.0 - params.beta2))?;
    let exp_avg = ((var.exp_avg.as_tensor() * params.beta2)? + grad_beta2)?;
    var.exp_avg.set(&exp_avg)?;

    Ok(())
}

impl Optimizer for Lion {
    type Config = ParamLion;

    fn new(vars: Vec<Var>, params: Self::Config) -> candle::Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let exp_avg = Var::zeros(shape, dtype, &device)?;
                Ok(VarLion {
                    var: var.into(),
                    exp_avg,
                })
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
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
                Ok(VarLion {
                    var,
                    exp_avg: Var::from_tensor(&exp_avg)?,
                })
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
        })
    }

    fn step(&mut self, grads: &GradStore) -> candle::Result<()> {
        let _backprop = disable_backprop();
        for var in &self.vars {
            let theta = &var.var;
            if let Some(g) = grads.get(theta) {
                lion_update(var, g, &self.params)?;
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
}
