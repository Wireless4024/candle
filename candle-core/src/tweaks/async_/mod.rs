use crate::{CpuStorage, DType, Result, Shape};

#[allow(dead_code)]
pub trait AsyncBackend {
    type Storage;
    async fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage_owned(s.clone()).await
    }
    async fn storage_from_cpu_storage_owned(&self, s: CpuStorage) -> Result<Self::Storage>;
    async fn storage_from_slice_async<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage>;
    async fn allocate_uninit_async(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage>;
    async fn zeros_impl_async(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage>;
    async fn synchronize_async(&self) -> Result<()>;
    async fn rand_uniform_async(&self, shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<Self::Storage>;
}
