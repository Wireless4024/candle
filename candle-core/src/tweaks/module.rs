use crate::{Module, ModuleT, Tensor};

/// More optimized Box<dyn Module>, this doesn't box the value if it useless and vtable is inlined
pub enum DynModule {
    Function(fn(&Tensor, bool) -> crate::Result<Tensor>),
    Module(*mut (), VTable),
}

unsafe impl Send for DynModule {}
unsafe impl Sync for DynModule {}

impl DynModule {
    #[inline]
    fn new<M: ModuleT>(module: M) -> Self {
        let vtable = VTable {
            forward: |ptr, input, training| unsafe { (&*ptr.cast::<M>()).forward_t(input, training) },
            drop: |ptr| unsafe {
                let _ = Box::from_raw(ptr.cast::<M>());
            },
        };
        let ptr = Box::into_raw(Box::new(module)).cast::<()>();
        DynModule::Module(ptr, vtable)
    }

    #[inline]
    pub fn forward(&self, input: &Tensor) -> crate::Result<Tensor> {
        self.forward_t(input, false)
    }

    pub fn forward_t(&self, input: &Tensor, training: bool) -> crate::Result<Tensor> {
        match self {
            DynModule::Function(f) => f(input, training),
            DynModule::Module(ptr, vtable) => (vtable.forward)(*ptr, input, training),
        }
    }
}

impl<M: Module> From<M> for DynModule {
    #[inline]
    default fn from(value: M) -> Self {
        DynModule::new(value)
    }
}

impl From<fn(&Tensor, bool) -> crate::Result<Tensor>> for DynModule {
    #[inline]
    fn from(value: fn(&Tensor, bool) -> crate::Result<Tensor>) -> Self {
        DynModule::Function(value)
    }
}

impl Drop for DynModule {
    fn drop(&mut self) {
        match self {
            DynModule::Function(_) => {}
            DynModule::Module(ptr, vtable) => (vtable.drop)(*ptr),
        }
    }
}

struct VTable {
    forward: fn(*mut (), &Tensor, bool) -> crate::Result<Tensor>,
    drop: fn(*mut ()),
}
