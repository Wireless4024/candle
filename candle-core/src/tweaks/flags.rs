use std::{cell::Cell, marker::PhantomData};

thread_local! {
    static IS_TRAINING: Cell<bool> = const { Cell::new(true) };
    static ENABLE_BACK_PROP: Cell<bool> = const { Cell::new(true) };
}

pub struct WhileTraining(PhantomData<*const ()>);
impl Drop for WhileTraining {
    fn drop(&mut self) {
        IS_TRAINING.set(false);
    }
}
pub fn set_training(training: bool) -> WhileTraining {
    IS_TRAINING.set(training);
    WhileTraining(PhantomData)
}

pub fn is_training() -> bool {
    IS_TRAINING.get()
}

pub struct DisabledBackprop(PhantomData<*const ()>);
impl Drop for DisabledBackprop {
    fn drop(&mut self) {
        ENABLE_BACK_PROP.set(true);
    }
}

pub fn disable_backprop() -> DisabledBackprop {
    ENABLE_BACK_PROP.set(false);
    DisabledBackprop(PhantomData)
}

pub fn backprop_enabled() -> bool {
    ENABLE_BACK_PROP.get()
}
