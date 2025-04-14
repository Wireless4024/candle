use crate::{cpu_backend::unary_map, CpuStorage, DType, Layout};
use half::{bf16, f16};

pub trait ToDTypeHelper: Copy {
    fn into_u8(self) -> u8;
    fn into_u16(self) -> u16;
    fn into_u32(self) -> u32;
    fn into_i64(self) -> i64;
    fn into_f16(self) -> f16;
    fn into_bf16(self) -> bf16;
    fn into_f32(self) -> f32;
    fn into_f64(self) -> f64;
}

macro_rules! primitive_convert {
    ($ty:ty) => {
        impl ToDTypeHelper for $ty {
            #[inline(always)]
            fn into_u8(self) -> u8 {
                self as _
            }
            #[inline(always)]
            fn into_u16(self) -> u16 {
                self as _
            }
            #[inline(always)]
            fn into_u32(self) -> u32 {
                self as _
            }
            #[inline(always)]
            fn into_i64(self) -> i64 {
                self as _
            }
            #[inline(always)]
            fn into_f16(self) -> f16 {
                f16::from_f32(self as f32)
            }
            #[inline(always)]
            fn into_bf16(self) -> bf16 {
                bf16::from_f32(self as f32)
            }
            #[inline(always)]
            fn into_f32(self) -> f32 {
                self as _
            }
            #[inline(always)]
            fn into_f64(self) -> f64 {
                self as _
            }
        }
    };
}

primitive_convert!(u8);
primitive_convert!(u16);
primitive_convert!(u32);
primitive_convert!(i64);
primitive_convert!(f32);
primitive_convert!(f64);

macro_rules! fp_to_int_cast {
    () => {
        #[inline(always)]
        fn into_u8(self) -> u8 {
            self.into_u32() as _
        }

        #[inline(always)]
        fn into_u16(self) -> u16 {
            self.into_u32() as _
        }

        #[inline(always)]
        fn into_u32(self) -> u32 {
            self.to_f32() as u32
        }

        #[inline(always)]
        fn into_i64(self) -> i64 {
            self.to_f32() as i64
        }
    };
}

impl ToDTypeHelper for f16 {
    fp_to_int_cast!();
    #[inline(always)]
    fn into_f16(self) -> f16 {
        self
    }

    #[inline(always)]
    fn into_bf16(self) -> bf16 {
        bf16::from_f32(self.to_f32())
    }

    #[inline(always)]
    fn into_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline(always)]
    fn into_f64(self) -> f64 {
        self.to_f64()
    }
}
impl ToDTypeHelper for bf16 {
    fp_to_int_cast!();
    #[inline(always)]
    fn into_f16(self) -> f16 {
        f16::from_f32(self.to_f32())
    }

    #[inline(always)]
    fn into_bf16(self) -> bf16 {
        self
    }

    #[inline(always)]
    fn into_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline(always)]
    fn into_f64(self) -> f64 {
        self.to_f64()
    }
}

pub fn unary_convert<T: ToDTypeHelper>(vs: &[T], layout: &Layout, dtype: DType) -> CpuStorage {
    match dtype {
        DType::U8 => CpuStorage::U8(unary_map(vs, layout, |v| v.into_u8())),
        DType::U16 => CpuStorage::U16(unary_map(vs, layout, |v| v.into_u16())),
        DType::U32 => CpuStorage::U32(unary_map(vs, layout, |v| v.into_u32())),
        DType::I64 => CpuStorage::I64(unary_map(vs, layout, |v| v.into_i64())),
        DType::BF16 => CpuStorage::BF16(unary_map(vs, layout, |v| v.into_bf16())),
        DType::F16 => CpuStorage::F16(unary_map(vs, layout, |v| v.into_f16())),
        DType::F32 => CpuStorage::F32(unary_map(vs, layout, |v| v.into_f32())),
        DType::F64 => CpuStorage::F64(unary_map(vs, layout, |v| v.into_f64())),
    }
}
