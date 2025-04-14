use crate::Tensor;

pub trait ParameterCount {
    fn parameter_count(&self) -> usize;
}

impl<T> ParameterCount for T {
    default fn parameter_count(&self) -> usize {
        0
    }
}

impl ParameterCount for Tensor {
    fn parameter_count(&self) -> usize {
        self.dims().iter().product::<usize>()
    }
}

impl<T: ParameterCount> ParameterCount for Option<T> {
    #[inline]
    fn parameter_count(&self) -> usize {
        match self {
            None => 0,
            Some(this) => this.parameter_count(),
        }
    }
}

impl<T: ParameterCount> ParameterCount for &[T] {
    #[inline]
    fn parameter_count(&self) -> usize {
        self.iter().map(|t| t.parameter_count()).sum()
    }
}

impl<T: ParameterCount> ParameterCount for Vec<T> {
    #[inline]
    fn parameter_count(&self) -> usize {
        self.as_slice().parameter_count()
    }
}

impl<T: ParameterCount> ParameterCount for Box<[T]> {
    #[inline]
    fn parameter_count(&self) -> usize {
        (&**self).parameter_count()
    }
}
