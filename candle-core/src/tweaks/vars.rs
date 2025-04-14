use crate::{
    tweaks::{ArcStr, ToArcString},
    Var,
};
use std::ops::{Deref, DerefMut};

#[derive(Clone)]
pub struct NamedVar {
    name: ArcStr,
    var: Var,
}

impl std::fmt::Display for NamedVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.name, self.var)
    }
}

impl std::fmt::Debug for NamedVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {:?}", self.name, self.var)
    }
}

impl Deref for NamedVar {
    type Target = Var;
    fn deref(&self) -> &Self::Target {
        &self.var
    }
}

impl DerefMut for NamedVar {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.var
    }
}

impl AsRef<str> for NamedVar {
    fn as_ref(&self) -> &str {
        &self.name
    }
}

impl<T: ToArcString> From<(T, Var)> for NamedVar {
    #[inline(always)]
    fn from((name, var): (T, Var)) -> Self {
        Self::new(name.to_string(), var)
    }
}

impl NamedVar {
    #[inline(always)]
    pub fn new(name: ArcStr, var: Var) -> Self {
        Self { name, var }
    }

    #[inline(always)]
    pub const fn name(&self) -> &ArcStr {
        &self.name
    }

    #[inline(always)]
    pub const fn var(&self) -> &Var {
        &self.var
    }

    pub fn into_var(self) -> Var {
        self.var
    }
}
