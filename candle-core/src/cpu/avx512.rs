use super::Cpu;
use core::arch::x86_64::*;

const STEP: usize = 64;
/// elements per register
const EPR: usize = 16;
/// array length
const ARR: usize = STEP / EPR;

pub struct CurrentCpu {}
impl Cpu<ARR> for CurrentCpu {
    type Unit = __m512;
    type Array = [__m512; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;

    #[inline(always)]
    fn n() -> usize {
        ARR
    }

    #[inline(always)]
    unsafe fn zero() -> Self::Unit {
        _mm512_setzero_ps()
    }

    #[inline(always)]
    unsafe fn zero_array() -> Self::Array {
        [_mm512_setzero_ps(); ARR]
    }

    #[inline(always)]
    unsafe fn load(mem_addr: *const f32) -> Self::Unit {
        _mm512_loadu_ps(mem_addr)
    }

    #[inline(always)]
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        _mm512_add_ps(a, b)
    }

    #[inline(always)]
    unsafe fn vec_fma(add: Self::Unit, mul_a: Self::Unit, mul_b: Self::Unit) -> Self::Unit {
        _mm512_fmadd_ps(mul_a, mul_b, add)
    }

    #[inline(always)]
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32) {
        let one = _mm512_add_ps(x[0], x[1]);
        let two = _mm512_add_ps(x[2], x[3]);
        // let three = _mm512_add_ps(x[4], x[5]);
        // let four = _mm512_add_ps(x[6], x[7]);
        let one_two = _mm512_add_ps(one, two);
        // let three_four = _mm512_add_ps(three, four);
        // let one_two_three_four = _mm512_add_ps(one_two, three_four);
        *y = _mm512_reduce_add_ps(one_two);
    }

    #[inline(always)]
    unsafe fn from_f32(v: f32) -> Self::Unit {
        _mm512_set1_ps(v)
    }

    #[inline(always)]
    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit) {
        _mm512_storeu_ps(mem_addr, a);
    }
}

// AVX‑512, half‑precision (16bit) lanes
// const HALF_STEP: usize = 32;
// /// elements per register
// const HALF_EPR: usize = 4;
// /// array length
// const HALF_ARR: usize = HALF_STEP / HALF_EPR;

// pub struct CurrentCpuF16 {}
// #[cfg(target_feature = "avx512fp16")]
// impl CpuF16<HALF_ARR> for CurrentCpuF16 {
//     type Unit = __m512h;
//     type Array = [__m512h; HALF_ARR];
//
//     const STEP: usize = HALF_STEP;
//     const EPR: usize = HALF_EPR;
//     #[inline(always)]
//     fn n() -> usize {
//         HALF_ARR
//     }
//     #[inline(always)]
//     unsafe fn zero() -> Self::Unit {
//         _mm512_setzero_ph()
//     }
//     #[inline(always)]
//     unsafe fn zero_array() -> Self::Array {
//         [Self::zero(); HALF_ARR]
//     }
//     #[inline(always)]
//     unsafe fn load(mem_addr: *const f16) -> Self::Unit {
//         _mm512_loadu_ph(mem_addr as *const _)
//     }
//     #[inline(always)]
//     unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
//         _mm512_add_ph(a, b)
//     }
//     #[inline(never)]
//     unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
//         _mm512_add_ph(_mm512_mul_ph(b, c), a)
//     }
//     #[inline(always)]
//     unsafe fn vec_reduce(x: Self::Array, y: *mut f32) {
//         let one = _mm512_add_ph(x[0], x[1]);
//         let two = _mm512_add_ph(x[2], x[3]);
//         let three = _mm512_add_ph(x[4], x[5]);
//         let four = _mm512_add_ph(x[6], x[7]);
//         let one_two = _mm512_add_ph(one, two);
//         let three_four = _mm512_add_ph(three, four);
//         let all = _mm512_add_ph(one_two, three_four);
//         *y = _mm512_reduce_min_ph(all) as _;
//     }
//     #[inline(always)]
//     unsafe fn from_f32(v: f32) -> Self::Unit {
//         _mm512_set1_ph(v as _)
//     }
//     #[inline(always)]
//     unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit) {
//         _mm512_storeu_ph(mem_addr as *mut _, a)
//     }
// }

#[cfg(test)]
mod tests {
    use crate::cpu::{Cpu, CurrentCpu};

    static ROW_A: [f32; 512] = {
        let mut a = [0.0; 512];
        let mut pos = 0;
        while pos < 512 {
            a[pos] = pos as f32 / 512.;
            pos += 1;
        }
        a
    };
    static ROW_B: [f32; 512] = {
        let mut a = [0.0; 512];
        let mut pos = 0;
        while pos < 512 {
            a[pos] = pos as f32 / 512.;
            pos += 1;
        }
        a
    };
    #[test]
    fn test_dot() {
        #[inline(never)]
        pub(crate) unsafe fn vec_dot_f32_avx512(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
            let np = k & !(CurrentCpu::STEP - 1);

            let mut sum = CurrentCpu::zero_array();
            let mut ax = CurrentCpu::zero_array();
            let mut ay = CurrentCpu::zero_array();

            for i in (0..np).step_by(CurrentCpu::STEP) {
                for j in 0..CurrentCpu::n() {
                    ax[j] = CurrentCpu::load(a_row.add(i + j * CurrentCpu::EPR));
                    ay[j] = CurrentCpu::load(b_row.add(i + j * CurrentCpu::EPR));

                    sum[j] = CurrentCpu::vec_fma(sum[j], ax[j], ay[j]);
                }
            }

            CurrentCpu::vec_reduce(sum, c);

            // leftovers
            for i in np..k {
                *c += *a_row.add(i) * (*b_row.add(i));
            }
        }
        #[inline(never)]
        pub(crate) unsafe fn vec_dot_f32_avx2(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
            use crate::cpu::avx::CurrentCpu;
            let np = k & !(CurrentCpu::STEP - 1);

            let mut sum = CurrentCpu::zero_array();
            let mut ax = CurrentCpu::zero_array();
            let mut ay = CurrentCpu::zero_array();

            for i in (0..np).step_by(CurrentCpu::STEP) {
                for j in 0..CurrentCpu::n() {
                    ax[j] = CurrentCpu::load(a_row.add(i + j * CurrentCpu::EPR));
                    ay[j] = CurrentCpu::load(b_row.add(i + j * CurrentCpu::EPR));

                    sum[j] = CurrentCpu::vec_fma(sum[j], ax[j], ay[j]);
                }
            }

            CurrentCpu::vec_reduce(sum, c);

            // leftovers
            for i in np..k {
                let a = *a_row.add(i);
                let b = *b_row.add(i);
                *c = a.mul_add(b, *c);
            }
        }
        #[inline(never)]
        pub(crate) unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
            // leftovers
            for i in 0..k {
                let a = *a_row.add(i);
                let b = *b_row.add(i);
                *c = a.mul_add(b, *c);
            }
        }
        unsafe {
            let mut out_avx2 = 0.0;
            let mut out_avx512 = 0.0;
            let mut out_scalar = 0.0;
            vec_dot_f32_avx2(ROW_A.as_ptr(), ROW_B.as_ptr(), &mut out_avx2, 511);
            vec_dot_f32_avx512(ROW_A.as_ptr(), ROW_B.as_ptr(), &mut out_avx512, 511);
            vec_dot_f32(ROW_A.as_ptr(), ROW_B.as_ptr(), &mut out_scalar, 511);
            println!("avx2: {out_avx2} avx512: {out_avx512}, scalar: {out_scalar}");
            assert_eq!(out_avx2 as i64, out_scalar as i64);
            assert_eq!(out_avx512 as i64, out_scalar as i64);
        }
    }
}
