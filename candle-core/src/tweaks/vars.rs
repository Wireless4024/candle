use crate::{
    tweaks::{ArcStr, ToArcString, VariableKind::DecayDivider},
    Var,
};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
};

thread_local! {
    static VAR_KIND:RefCell<(VariableKind,bool)> = const { RefCell::new((VariableKind::Standard,false)) };
}

pub struct ResetDefaultOnDrop;
impl Drop for ResetDefaultOnDrop {
    fn drop(&mut self) {
        VAR_KIND.replace((VariableKind::Standard, false));
    }
}

pub fn with_var_kind(kind: VariableKind) -> ResetDefaultOnDrop {
    let (old, overridden) = VAR_KIND.with(|x| *x.borrow());
    if overridden && old == VariableKind::Standard {
        return ResetDefaultOnDrop;
    }
    VAR_KIND.replace((kind, true));
    ResetDefaultOnDrop
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum VariableKind {
    /// Prefer standard optimizer like AdamW
    Standard,
    /// Some optimizer like Muon have matrix-aware update
    GeometryAware,
    /// Do not apply weight decay to this variable,
    /// override weight decay by (u16::MAX/decay_divider)
    DecayDivider(u32),
    /// Do not optimize this variable
    NoOptimize,
}

impl VariableKind {
    #[allow(non_upper_case_globals)]
    pub const NoDecay: Self = DecayDivider(u16::MAX as _);
    #[allow(non_upper_case_globals)]
    pub const OriginalDecay: Self = DecayDivider(0);

    pub fn get_decay(&self, decay: f64) -> Option<f64> {
        if decay <= 0.0 {
            return None;
        }
        match *self {
            Self::Standard => Some(decay),
            Self::GeometryAware => Some(decay),
            Self::DecayDivider(divider) => {
                if divider == 0 {
                    Some(decay)
                } else if divider == u16::MAX as u32 {
                    None
                } else {
                    Some(decay * (u16::MAX as f64 / (divider as f64)))
                }
            }
            Self::NoOptimize => None,
        }
    }

    pub fn with_decay_multiplier(multiplier: f64) -> Self {
        if multiplier == 0.0 {
            Self::NoDecay
        } else {
            Self::DecayDivider((u16::MAX as f64 / multiplier).clamp(0.0, u32::MAX as f64) as u32)
        }
    }
}

impl Default for VariableKind {
    fn default() -> Self {
        VAR_KIND.with(|v| v.borrow().0)
    }
}

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

impl From<Var> for NamedVar {
    fn from(value: Var) -> Self {
        NamedVar::new(ArcStr::default(), value)
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
