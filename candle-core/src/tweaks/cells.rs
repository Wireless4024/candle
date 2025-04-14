use std::cell::RefCell;
use std::fmt::Debug;

pub struct RefCellWrap<T> {
    pub inner: RefCell<T>,
}

impl<T:Debug> Debug for RefCellWrap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl<T> RefCellWrap<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: RefCell::new(inner),
        }
    }

    #[inline(always)]
    pub fn lock(&self) -> Result<std::cell::RefMut<T>, ()> {
        Ok(self.inner.borrow_mut())
    }

    #[inline(always)]
    pub fn read(&self) -> Result<std::cell::Ref<T>, ()> {
        Ok(self.inner.borrow())
    }

    #[inline(always)]
    pub fn write(&self) -> Result<std::cell::RefMut<T>, ()> {
        Ok(self.inner.borrow_mut())
    }
}
