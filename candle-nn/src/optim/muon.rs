use candle::{Tensor, Var};
use candle::backprop::GradStore;
use crate::Optimizer;

#[derive(Clone, Copy, Debug)]
pub struct ParamsMuon {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub ns_steps: usize,
    pub nesterov: bool,
}

impl Default for ParamsMuon {
    #[inline]
    fn default() -> Self {
        Self {
            lr: 0.001,
            momentum: 0.95,
            weight_decay: 0.001,
            ns_steps: 5,
            nesterov: true,
        }
    }
}

struct VarMuon {
    var: candle::tweaks::NamedVar,
    momentum: Var,
}

/// https://kellerjordan.github.io/posts/muon/
/// Muon is an optimizer for the hidden weights of a neural network. Other parameters, such as embeddings, classifier
/// heads, and hidden gains/biases should be optimized using standard AdamW. Muon should be used as follows:
pub struct Muon {
    vars: Vec<VarMuon>,
    params: ParamsMuon,
}

/// Performs the Newton-Schulz iteration to orthogonalize the input matrix `g`.
/// This is a quintic iteration designed for efficiency and stability.
fn zeropower_via_newtonschulz5(g: &Tensor, steps: usize) -> candle::Result<Tensor> {
    if g.rank() < 2 {
        return Ok(g.clone());
    }

    let original_shape = g.shape();
    let (h, w) = (
        original_shape.dims()[original_shape.rank() - 2],
        original_shape.dims()[original_shape.rank() - 1],
    );

    let mut x = g.to_dtype(candle::DType::F32)?;
    let mut transpose = false;

    if h > w {
        x = x.t()?;
        transpose = true;
    }

    // Normalize by the Frobenius norm (proxy for spectral norm)
    let norm = x.norm()?.broadcast_as(x.shape())?;
    x = x.div(&(norm + 1e-7)?)?;

    // Quintic iteration coefficients
    let a = 3.4445f64;
    let b = -4.7750f64;
    let c = 2.0315f64;

    for _ in 0..steps {
        let xt = x.t()?;
        let a_mat = x.matmul(&xt)?;
        let b_mat = ((&a_mat * b)? + a_mat.matmul(&a_mat)? * c)?;
        x = ((&x * a)? + b_mat.matmul(&x)?)?;
    }

    if transpose {
        x = x.t()?;
    }

    x.to_dtype(g.dtype())
}

/// Calculates the Muon update for a given gradient and momentum buffer.
fn muon_update(grad: &Tensor, momentum_buffer: &Var, param: &ParamsMuon) -> candle::Result<Tensor> {
    // Update momentum buffer: momentum.lerp_(grad, 1 - beta)
    let new_momentum = ((momentum_buffer.as_tensor() * param.momentum)? + (grad * (1.0 - param.momentum))?)?;
    momentum_buffer.set(&new_momentum)?;

    // Calculate update tensor
    let mut update = if param.nesterov {
        // update = grad.lerp_(momentum, beta)
        ((grad * (1.0 - param.momentum))? + (momentum_buffer.as_tensor() * param.momentum)?)?
    } else {
        momentum_buffer.as_tensor().clone()
    };

    let original_shape = update.shape().clone();
    let dims = original_shape.dims();

    // Reshape conv filters (4D) to be 2D for orthogonalization
    if update.rank() == 4 {
        update = update.flatten_from(1)?;
    }

    // Orthogonalize the update
    update = zeropower_via_newtonschulz5(&update, param.ns_steps)?;

    // Reshape back to original shape
    update = update.reshape(original_shape.clone())?;

    // Apply scaling factor
    let h = dims.get(dims.len() - 2).cloned().unwrap_or(1) as f64;
    let w = dims.last().cloned().unwrap_or(1) as f64;
    let scale = (h / w).max(1.0).sqrt();
    update = (update * scale)?;

    Ok(update)
}

impl Optimizer for Muon {
    type Config = ParamsMuon;

    fn new(vars: Vec<Var>, config: Self::Config) -> candle::Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let momentum = Var::zeros(shape, dtype, device)?;
                Ok(VarMuon {
                    var: var.into(),
                    momentum
                })
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params: config,
        })
    }

    fn new2(vars: Vec<candle::tweaks::NamedVar>, params: Self::Config, vb: crate::VarBuilder) -> candle::Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let vb = vb.pp(var.name());
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let velocity = vb
                    .get_with_hints_dtype(shape, "velocity", crate::Init::Const(0.0), dtype)?
                    .to_device(device)?;
                Ok(VarMuon {
                    var,
                    momentum: Var::from_tensor(&velocity)?,
                })
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
        })
    }

    fn step(&mut self, grads: &GradStore) -> candle::Result<()> {
        let lr = self.params.lr;

        for var in self.vars.iter() {
            if let Some(grad) = grads.get(&var.var) {
                // Apply weight decay
                if self.params.weight_decay > 0.0 {
                    let decay = ((var.var.as_tensor() * lr)? * self.params.weight_decay)?;
                    var.var.set(&var.var.sub(&decay)?)?;
                }

                // Calculate the orthogonalized update
                let update = muon_update(grad, &var.momentum, &self.params)?;

                // Apply the final update
                var.var.set(&var.var.sub(&(update * lr)?)?)?;
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