use candle::{DType, Tensor};

#[derive(Clone, Copy, Debug, Default)]
pub enum GradientClipping {
    #[default]
    None,
    Scale {
        max: f64,
    },
    Clamp {
        min: f64,
        max: f64,
    },
}

impl GradientClipping {
    pub fn forward(&self, xs: &Tensor, name: &str) -> candle::Result<Tensor> {
        match *self {
            GradientClipping::Scale { max } => {
                //let sq_sum_norm = xs.sqr()?.sum_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?;
                let grad_norm = xs.max_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?;
                let eps = 1e-12f64;
                //let grad_norm = sq_sum_norm.sqrt();
                if grad_norm > max {
                    println!("clamping gradient {grad_norm} > {max} ({name})");
                    xs * (max / (grad_norm + eps))
                } else {
                    Ok(xs.clone())
                }
            }
            GradientClipping::Clamp { min, max } => xs.clamp(min, max),
            GradientClipping::None => Ok(xs.clone()),
        }
    }
}
