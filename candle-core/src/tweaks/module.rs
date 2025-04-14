use crate::{tweaks::ParameterCount, ModuleT, Tensor};
use std::{any::TypeId, fmt::Debug, mem::MaybeUninit};

/// More optimized Box<dyn Module>, this doesn't box the value if it useless and vtable is inlined
#[expect(private_interfaces)]
pub enum DynModule {
    Function(fn(&Tensor, bool) -> crate::Result<Tensor>),
    Module(*mut (), VTable),
}

// I have no idea how this work, this should be 32 bytes so it can do simd memcopy but idk why it 24
const _: () = assert!(size_of::<DynModule>() == size_of::<usize>() * 3);

unsafe impl Send for DynModule {}
unsafe impl Sync for DynModule {}

impl DynModule {
    #[inline]
    pub fn new<M: ModuleT + 'static>(module: M) -> Self {
        let vtable = VTable {
            forward: |ptr, input, training| unsafe { (&*ptr.cast::<M>()).forward_t(input, training) },
            extra: &VTableExtra {
                type_id: || TypeId::of::<M>(),
                type_name: || std::any::type_name::<M>(),
                parameter_count: |ptr| unsafe { &*ptr.cast::<M>() }.parameter_count(),
                drop: |ptr| unsafe {
                    let _ = Box::from_raw(ptr.cast::<M>());
                },
            },
        };
        let ptr = Box::into_raw(Box::new(module)).cast::<()>();
        DynModule::Module(ptr, vtable)
    }

    #[inline]
    pub const fn new_fn(f: fn(&Tensor, bool) -> crate::Result<Tensor>) -> Self {
        DynModule::Function(f)
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

    pub fn downcast_ref<M: ModuleT + 'static>(&self) -> Option<&M> {
        match self {
            DynModule::Function(_) => None,
            DynModule::Module(ptr, vtable) => {
                if (vtable.extra.type_id)() == TypeId::of::<M>() {
                    Some(unsafe { &*ptr.cast::<M>() })
                } else {
                    None
                }
            }
        }
    }

    pub fn downcast_mut<M: ModuleT + 'static>(&mut self) -> Option<&mut M> {
        match self {
            DynModule::Function(_) => None,
            DynModule::Module(ptr, vtable) => {
                if (vtable.extra.type_id)() == TypeId::of::<M>() {
                    Some(unsafe { &mut *ptr.cast::<M>() })
                } else {
                    None
                }
            }
        }
    }

    pub fn downcast<T: 'static>(mut self) -> Result<T, Self> {
        match &mut self {
            DynModule::Function(_) => Err(self),
            DynModule::Module(ptr, vtable) => {
                if (vtable.extra.type_id)() == TypeId::of::<T>() {
                    // move value from heap to stack
                    let value = unsafe { ptr.cast::<T>().read() };
                    // free pointer
                    let _ = unsafe { Box::from_raw(ptr.cast::<MaybeUninit<T>>()) };
                    *ptr = std::ptr::null_mut();
                    Ok(value)
                } else {
                    Err(self)
                }
            }
        }
    }
}

impl<M: ModuleT + 'static> From<M> for DynModule {
    #[inline]
    default fn from(value: M) -> Self {
        DynModule::new(value)
    }
}

impl From<fn(&Tensor, bool) -> crate::Result<Tensor>> for DynModule {
    #[inline]
    fn from(value: fn(&Tensor, bool) -> crate::Result<Tensor>) -> Self {
        DynModule::new_fn(value)
    }
}

impl Debug for DynModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DynModule::Function(_) => write!(f, "Function"),
            DynModule::Module(_, vtable) => write!(f, "Module({})", (vtable.extra.type_name)()),
        }
    }
}

impl Drop for DynModule {
    fn drop(&mut self) {
        match self {
            DynModule::Function(_) => {}
            DynModule::Module(ptr, vtable) => {
                if ptr.is_null() {
                    return;
                }
                (vtable.extra.drop)(*ptr)
            }
        }
    }
}

// fast vtable, forward function is inlined
struct VTable {
    forward: fn(*mut (), &Tensor, bool) -> crate::Result<Tensor>,
    extra: &'static VTableExtra,
}

// nested vtable
struct VTableExtra {
    type_id: fn() -> TypeId,
    type_name: fn() -> &'static str,
    parameter_count: fn(*const ()) -> usize,
    drop: fn(*mut ()),
}
