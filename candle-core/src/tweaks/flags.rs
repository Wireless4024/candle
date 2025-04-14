use std::{cell::Cell, marker::PhantomData};

thread_local! {
    static IS_TRAINING: Cell<bool> = const { Cell::new(true) };
    static ENABLE_BACK_PROP: Cell<bool> = const { Cell::new(true) };
}

pub struct WhileTraining(PhantomData<*const ()>, bool);
impl Drop for WhileTraining {
    fn drop(&mut self) {
        IS_TRAINING.set(self.1);
    }
}
pub fn set_training(training: bool) -> WhileTraining {
    let last = IS_TRAINING.get();
    IS_TRAINING.set(training);
    WhileTraining(PhantomData, last)
}

pub fn is_training() -> bool {
    IS_TRAINING.get()
}

pub struct DisabledBackprop(PhantomData<*const ()>, bool);
impl Drop for DisabledBackprop {
    fn drop(&mut self) {
        ENABLE_BACK_PROP.set(self.1);
    }
}

pub fn disable_backprop() -> DisabledBackprop {
    let last = ENABLE_BACK_PROP.get();
    ENABLE_BACK_PROP.set(false);
    DisabledBackprop(PhantomData, last)
}

pub fn backprop_enabled() -> bool {
    ENABLE_BACK_PROP.get()
}
