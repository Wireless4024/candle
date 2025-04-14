use std::cell::RefCell;

pub struct RefCellWrap<T> {
    pub inner: RefCell<T>,
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
