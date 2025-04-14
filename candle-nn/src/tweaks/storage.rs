use crate::{
    tweaks::{LoadParam, ModuleRegistry, SerializableModule},
    var_builder::SimpleBackend,
    Init, VarBuilder,
};
use ahash::HashMap;
use candle::{
    safetensors::Load,
    tweaks::{ArcStr, NamedVar, RefCellWrap, ToArcString},
    DType, Device, Error, Shape, Tensor, TensorId, Var,
};
use safetensors::{serialize_to_file, tensor::TensorView, Dtype, View};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, fmt::Display, path::Path, rc::Rc, str::FromStr};

#[derive(Clone, Debug)]
pub struct SafeTensorStorage {
    inner: Rc<RefCellWrap<SafeTensorStorageInner>>,
}

impl Default for SafeTensorStorage {
    fn default() -> Self {
        Self {
            inner: Rc::new(RefCellWrap::new(SafeTensorStorageInner::new())),
        }
    }
}

impl SafeTensorStorage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_vb(path: impl AsRef<Path>, dtype: DType, device: &Device) -> candle::Result<VarBuilder<'static>> {
        let this = Self::load(path, device)?;
        Ok(this.vb(dtype, device.clone()))
    }

    pub fn load(path: impl AsRef<Path>, device: &Device) -> candle::Result<Self> {
        let this = Self::new();
        this.inner.write().unwrap().load(path.as_ref(), device)?;
        Ok(this)
    }

    pub fn load_from(&self, path: impl AsRef<Path>, device: &Device) -> candle::Result<()> {
        let mut inner = self.inner.write().unwrap();
        inner.load(path.as_ref(), device)?;
        Ok(())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> candle::Result<()> {
        let inner = self.inner.read().unwrap();
        inner.save(path.as_ref())
    }

    pub fn clone_from(&self, other: &Self) {
        self.clone_from_map_key(other, |k| k.clone(), |k| k.clone())
    }

    pub fn clone_from_map_key<K1: ToArcString, K2: ToArcString>(
        &self,
        other: &Self,
        map_tensor: impl Fn(&ArcStr) -> K1,
        map_info: impl Fn(&ArcStr) -> K2,
    ) {
        let mut inner = self.inner.write().unwrap();
        inner.clone_from(&other.inner.read().unwrap(), map_tensor, map_info)
    }

    pub fn all_vars(&self) -> Vec<NamedVar> {
        let inner = self.inner.read().unwrap();
        inner.all_vars(|_| true)
    }

    pub fn all_vars_prefix(&self, prefix: &str) -> Vec<NamedVar> {
        self.all_vars_filter(|name| name.starts_with(prefix))
    }

    pub fn all_vars_filter<F: FnMut(&str) -> bool>(&self, filter: F) -> Vec<NamedVar> {
        let inner = self.inner.read().unwrap();
        inner.all_vars(filter)
    }

    pub fn bucket_select<F: FnMut(&str, &Var) -> usize>(&self, select: F) -> Vec<Vec<NamedVar>> {
        let inner = self.inner.read().unwrap();
        inner.bucket_select(select)
    }

    pub fn vb(&self, dtype: DType, device: Device) -> VarBuilder<'static> {
        VarBuilder::from_backend(Box::new(self.clone()), dtype, device)
    }

    pub fn model_vb(&self, device: Device) -> VarBuilder<'static> {
        self.vb(self.get_default_dtype().unwrap_or(DType::F32), device)
            .pp("model")
    }

    pub fn set_default_dtype(&self, dtype: DType) {
        self.put_meta("dtype", dtype.as_str());
    }

    pub fn get_default_dtype(&self) -> candle::Result<DType> {
        let Some(dtype) = self.get_meta("dtype") else {
            return Ok(DType::F32);
        };
        DType::from_str(&dtype).map_err(Error::wrap)
    }

    pub fn store_model_schema<M: SerializableModule>(&self, model: &M) {
        let cfg = model.config_value("model");
        let json = serde_json::to_string(&cfg).unwrap();
        self.put_meta("model", &json);
    }

    pub fn load_model_schema<M: SerializableModule>(&self, reg: &ModuleRegistry, device: Device) -> candle::Result<M> {
        let Some(model) = self.get_meta("model") else {
            return Err(Error::msg("No model found in storage"));
        };
        let cfg: LoadParam = serde_json::from_str(&model).map_err(Error::wrap)?;
        let vb = self.model_vb(device);
        let module = reg.load(cfg, &vb)?;
        let module_name = format!("{:?}", module);
        let Ok(module) = module.downcast::<M>() else {
            return Err(Error::msg(format!(
                "Invalid module type, casting {module_name:?} to {}",
                std::any::type_name::<M>()
            )));
        };
        Ok(module)
    }

    pub fn put_meta(&self, key: impl ToArcString, value: impl ToArcString) {
        let mut inner = self.inner.write().unwrap();
        inner.info.insert(key.to_string(), value.to_string());
    }

    pub fn put_meta_display(&self, key: impl ToArcString, value: impl Display) {
        self.put_meta(key, value.to_string());
    }

    pub fn get_meta(&self, key: &str) -> Option<ArcStr> {
        let inner = self.inner.read().unwrap();
        inner.info.get(key).cloned()
    }

    pub fn get_meta_parse<S: FromStr>(&self, key: &str) -> candle::Result<S>
    where
        S::Err: Display + Send + Sync + 'static,
    {
        let value = self
            .get_meta(key)
            .ok_or_else(|| Error::msg(format!("No meta value for {}", key)))?;
        value.parse().map_err(Error::wrap)
    }

    pub fn get_primitive<T: TryFrom<Primitive>>(&self, name: &str) -> candle::Result<Option<T>>
    where
        T::Error: Display,
    {
        let inner = self.inner.read().unwrap();
        inner.get_primitive(name)
    }

    pub fn put_primitive(&self, name: &str, value: impl Into<Primitive>) {
        let mut inner = self.inner.write().unwrap();
        inner.put_primitive(name, value);
    }

    pub fn get_by_id(&self, id: TensorId) -> Option<(ArcStr, Tensor)> {
        let inner = self.inner.read().unwrap();
        inner
            .map
            .iter()
            .filter_map(|(name, tensor)| Some(name.clone()).zip(tensor.as_tensor().ok().cloned()))
            .find(|(_, tensor)| tensor.id() == id)
    }
}

#[derive(Debug)]
struct SafeTensorStorageInner {
    map: HashMap<ArcStr, TensorWrapper>,
    info: HashMap<ArcStr, ArcStr>,
}

impl SafeTensorStorageInner {
    pub fn new() -> Self {
        Self {
            map: HashMap::default(),
            info: HashMap::default(),
        }
    }

    pub fn clone_from<K1: ToArcString, K2: ToArcString>(
        &mut self,
        other: &Self,
        map_tensor: impl Fn(&ArcStr) -> K1,
        map_info: impl Fn(&ArcStr) -> K2,
    ) {
        for (k, v) in &other.map {
            self.map.insert(map_tensor(k).to_string(), v.clone());
        }
        for (k, v) in &other.info {
            self.info.insert(map_info(k).to_string(), v.clone());
        }
    }

    pub fn save(&self, path: &Path) -> candle::Result<()> {
        let mut info = self
            .info
            .iter()
            .map(|(k, v)| (k.cvt_string(), v.cvt_string()))
            .collect::<std::collections::HashMap<_, _>>();
        let mut kinds = HashMap::default();
        for (key, value) in &self.map {
            kinds.insert(key.as_str(), value.kind());
        }
        let kinds = serde_json::to_string(&kinds).map_err(Error::wrap)?;
        info.insert("__kinds__".into(), kinds);
        let info = Some(info);
        serialize_to_file(self.map.iter().map(|(k, v)| (k, v.make_le())), info, path).map_err(Error::wrap)?;
        Ok(())
    }

    pub fn load(&mut self, path: &Path, device: &Device) -> candle::Result<()> {
        let data = unsafe { candle::safetensors::MmapedSafetensors::new(path)? };
        for (key, value) in data.metadata() {
            self.info.insert(key.into(), value.into());
        }
        let mut kinds = HashMap::default();
        if let Some(kinds_str) = self.info.get("__kinds__") {
            kinds = serde_json::from_str::<HashMap<String, TensorViewKind>>(kinds_str).map_err(Error::wrap)?;
        }
        for (name, view) in data.tensors() {
            let kind = kinds.get(&name).unwrap_or(&TensorViewKind::Tensor);
            let item = kind.load(view, device).map_err(Error::wrap)?;
            self.map.insert(name.into(), item);
        }
        Ok(())
    }

    pub fn all_vars(&self, mut filter: impl FnMut(&str) -> bool) -> Vec<NamedVar> {
        self.map
            .iter()
            .filter(|(k, _)| filter(k.as_str()))
            .filter_map(|(k, v)| Some(NamedVar::new(k.clone(), v.as_var().cloned().ok()?)))
            .collect()
    }

    pub fn bucket_select<F: FnMut(&str, &Var) -> usize>(&self, mut select: F) -> Vec<Vec<NamedVar>> {
        let mut buckets = Vec::<Vec<NamedVar>>::new();
        for (k, v) in self.map.iter() {
            if let Ok(var) = v.as_var() {
                let bucket = select(k.as_str(), var);
                if buckets.len() < bucket {
                    buckets.resize_with(bucket + 1, Vec::new);
                }
                buckets[bucket].push(NamedVar::new(k.clone(), v.as_var().cloned().unwrap()));
            }
        }
        buckets
    }

    pub fn copy(&self) -> candle::Result<Self> {
        let mut new_data = self.map.clone();
        let new_info = self.info.clone();
        for value in new_data.values_mut() {
            *value = value.copy()?;
        }
        Ok(Self {
            map: new_data,
            info: new_info,
        })
    }

    pub fn get<S: Into<Shape>>(
        &mut self,
        shape: S,
        path: &str,
        init: Init,
        dtype: DType,
        device: &Device,
    ) -> candle::Result<Tensor> {
        let shape = shape.into();
        if let Some(tensor) = self.map.get(path) {
            let tensor_shape = tensor.shape();
            if shape.dims() != tensor_shape {
                candle::bail!("shape mismatch on {path}: {shape:?} <> {tensor_shape:?}")
            }
            return tensor.as_tensor().cloned();
        }
        let var = init.var(shape, dtype, device)?;
        let tensor = var.as_tensor().clone();
        self.map.insert(path.into(), var.into());
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle::Result<Tensor> {
        let tensor = self.map.get(name).ok_or_else(|| {
            Error::CannotFindTensor {
                path: String::from(name),
            }
            .bt()
        })?;
        tensor.as_tensor()?.to_device(dev)?.to_dtype(dtype)
    }

    fn put_primitive(&mut self, name: &str, value: impl Into<Primitive>) {
        self.map.insert(name.into(), TensorWrapper::Primitive(value.into()));
    }

    fn get_primitive<T: TryFrom<Primitive>>(&self, name: &str) -> candle::Result<Option<T>>
    where
        T::Error: Display,
    {
        self.map
            .get(name)
            .map(|val| val.as_primitive().copied()?.try_into().map_err(Error::msg))
            .transpose()
    }
}

impl SimpleBackend for SafeTensorStorage {
    fn get(&self, s: Shape, name: &str, h: Init, dtype: DType, dev: &Device) -> candle::Result<Tensor> {
        let mut inner = self.inner.write().unwrap();
        inner.get(s, name, h, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle::Result<Tensor> {
        let inner = self.inner.write().unwrap();
        inner.get_unchecked(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.read().unwrap().map.contains_key(name)
    }

    fn put_tensor(&self, name: &str, tensor: Tensor, as_var: bool) -> candle::Result<Tensor> {
        if as_var {
            let var = Var::from_tensor(&tensor)?;
            let tensor = var.as_tensor().clone();
            self.inner.write().unwrap().map.insert(name.into(), var.into());
            Ok(tensor)
        } else {
            self.inner
                .write()
                .unwrap()
                .map
                .insert(name.into(), tensor.clone().into());
            Ok(tensor)
        }
    }
}

#[derive(Clone, Debug)]
enum TensorWrapper {
    Tensor(Tensor),
    Var(Var),
    Primitive(Primitive),
}

impl From<Tensor> for TensorWrapper {
    #[inline(always)]
    fn from(value: Tensor) -> Self {
        TensorWrapper::Tensor(value)
    }
}
impl From<Var> for TensorWrapper {
    #[inline(always)]
    fn from(value: Var) -> Self {
        TensorWrapper::Var(value)
    }
}
impl From<Primitive> for TensorWrapper {
    #[inline(always)]
    fn from(value: Primitive) -> Self {
        TensorWrapper::Primitive(value)
    }
}

impl TensorWrapper {
    pub fn copy(&self) -> candle::Result<Self> {
        Ok(match self {
            TensorWrapper::Tensor(t) => TensorWrapper::Tensor(t.copy_as_var(false)?),
            TensorWrapper::Var(v) => TensorWrapper::Var(Var::from_tensor(&v.as_tensor().copy_as_var(true)?)?),
            TensorWrapper::Primitive(p) => TensorWrapper::Primitive(*p),
        })
    }

    #[inline]
    pub fn kind(&self) -> TensorViewKind {
        match self {
            TensorWrapper::Tensor(_) => TensorViewKind::Tensor,
            TensorWrapper::Var(_) => TensorViewKind::Var,
            TensorWrapper::Primitive(p) => TensorViewKind::Primitive(p.kind()),
        }
    }

    pub fn make_le(&self) -> Self {
        match self {
            TensorWrapper::Tensor(t) => TensorWrapper::Tensor(t.clone()),
            TensorWrapper::Var(v) => TensorWrapper::Var(v.clone()),
            TensorWrapper::Primitive(p) => TensorWrapper::Primitive(p.make_le()),
        }
    }

    pub fn as_tensor(&self) -> candle::Result<&Tensor> {
        match self {
            TensorWrapper::Tensor(t) => Ok(t),
            TensorWrapper::Var(v) => Ok(v.as_tensor()),
            TensorWrapper::Primitive(_) => Err(Error::msg("Cannot convert primitive to tensor")),
        }
    }

    pub fn as_var(&self) -> candle::Result<&Var> {
        match self {
            TensorWrapper::Tensor(_) => Err(Error::msg("Cannot convert tensor to var")),
            TensorWrapper::Var(v) => Ok(v),
            TensorWrapper::Primitive(_) => Err(Error::msg("Cannot convert primitive to var")),
        }
    }

    pub fn as_primitive(&self) -> candle::Result<&Primitive> {
        match self {
            TensorWrapper::Tensor(_) => Err(Error::msg("Cannot convert tensor to primitive")),
            TensorWrapper::Var(_) => Err(Error::msg("Cannot convert var to primitive")),
            TensorWrapper::Primitive(p) => Ok(p),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
enum TensorViewKind {
    Tensor,
    Var,
    Primitive(PrimitiveKind),
}

impl TensorViewKind {
    pub fn load(&self, data: TensorView, device: &Device) -> candle::Result<TensorWrapper> {
        match self {
            TensorViewKind::Tensor => Ok(TensorWrapper::Tensor(data.load(device)?)),
            TensorViewKind::Var => Ok(TensorWrapper::Var(Var::from_tensor(&data.load(device)?)?)),
            TensorViewKind::Primitive(kind) => Ok(TensorWrapper::Primitive(kind.load(data)?)),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum PrimitiveKind {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl PrimitiveKind {
    fn load(&self, data: TensorView) -> candle::Result<Primitive> {
        let data = data.data();
        use PrimitiveKind::*;
        fn check_len<const LEN: usize>(data: &[u8]) -> candle::Result<[u8; LEN]> {
            if data.len() != LEN {
                return Err(Error::msg(format!("Expected {} bytes, got {}", LEN, data.len())));
            }
            let mut buf = [0u8; LEN];
            buf.copy_from_slice(data);
            Ok(buf)
        }
        match self {
            Bool => {
                let data = check_len::<1>(data)?;
                Ok(Primitive::Bool(data[0] != 0))
            }
            I8 => {
                let data = check_len::<1>(data)?;
                Ok(Primitive::I8(data[0] as _))
            }
            U8 => {
                let data = check_len::<1>(data)?;
                Ok(Primitive::U8(data[0]))
            }
            I16 => {
                let data = check_len::<2>(data)?;
                Ok(Primitive::I16(i16::from_le_bytes(data)))
            }
            U16 => {
                let data = check_len::<2>(data)?;
                Ok(Primitive::U16(u16::from_le_bytes(data)))
            }
            I32 => {
                let data = check_len::<4>(data)?;
                Ok(Primitive::I32(i32::from_le_bytes(data)))
            }
            U32 => {
                let data = check_len::<4>(data)?;
                Ok(Primitive::U32(u32::from_le_bytes(data)))
            }
            I64 => {
                let data = check_len::<8>(data)?;
                Ok(Primitive::I64(i64::from_le_bytes(data)))
            }
            U64 => {
                let data = check_len::<8>(data)?;
                Ok(Primitive::U64(u64::from_le_bytes(data)))
            }
            F32 => {
                let data = check_len::<4>(data)?;
                Ok(Primitive::F32(f32::from_le_bytes(data)))
            }
            F64 => {
                let data = check_len::<8>(data)?;
                Ok(Primitive::F64(f64::from_le_bytes(data)))
            }
        }
    }
}

impl Display for PrimitiveKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[inline]
fn dtype_candle_to_safetensors(dtype: DType) -> Dtype {
    match dtype {
        DType::U8 => Dtype::U8,
        DType::U16 => Dtype::U16,
        DType::U32 => Dtype::U32,
        DType::I64 => Dtype::I64,
        DType::BF16 => Dtype::BF16,
        DType::F16 => Dtype::F16,
        DType::F32 => Dtype::F32,
        DType::F64 => Dtype::F64,
        DType::F8E4M3 => Dtype::F8_E4M3,
    }
}

impl View for TensorWrapper {
    fn dtype(&self) -> Dtype {
        match self {
            TensorWrapper::Tensor(t) => dtype_candle_to_safetensors(t.dtype()),
            TensorWrapper::Var(v) => dtype_candle_to_safetensors(v.dtype()),
            TensorWrapper::Primitive(p) => p.dtype(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            TensorWrapper::Tensor(t) => View::shape(t),
            TensorWrapper::Var(v) => View::shape(v.as_tensor()),
            TensorWrapper::Primitive(p) => p.shape(),
        }
    }

    fn data(&self) -> Cow<'_, [u8]> {
        match self {
            TensorWrapper::Tensor(t) => t.data(),
            TensorWrapper::Var(v) => v.as_tensor().data(),
            TensorWrapper::Primitive(p) => p.data(),
        }
    }

    fn data_len(&self) -> usize {
        match self {
            TensorWrapper::Tensor(t) => t.data_len(),
            TensorWrapper::Var(v) => v.as_tensor().data_len(),
            TensorWrapper::Primitive(p) => p.data_len(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Primitive {
    Bool(bool),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
}

impl Primitive {
    #[inline]
    pub fn kind(&self) -> PrimitiveKind {
        match self {
            Primitive::Bool(_) => PrimitiveKind::Bool,
            Primitive::I8(_) => PrimitiveKind::I8,
            Primitive::U8(_) => PrimitiveKind::U8,
            Primitive::I16(_) => PrimitiveKind::I16,
            Primitive::U16(_) => PrimitiveKind::U16,
            Primitive::I32(_) => PrimitiveKind::I32,
            Primitive::U32(_) => PrimitiveKind::U32,
            Primitive::I64(_) => PrimitiveKind::I64,
            Primitive::U64(_) => PrimitiveKind::U64,
            Primitive::F32(_) => PrimitiveKind::F32,
            Primitive::F64(_) => PrimitiveKind::F64,
        }
    }

    pub fn make_le(self) -> Self {
        match self {
            Primitive::Bool(b) => Primitive::Bool(b),
            Primitive::I8(i) => Primitive::I8(i),
            Primitive::U8(u) => Primitive::U8(u),
            Primitive::I16(i) => Primitive::I16(i.to_le()),
            Primitive::U16(u) => Primitive::U16(u.to_le()),
            Primitive::I32(i) => Primitive::I32(i.to_le()),
            Primitive::U32(u) => Primitive::U32(u.to_le()),
            Primitive::I64(i) => Primitive::I64(i.to_le()),
            Primitive::U64(u) => Primitive::U64(u.to_le()),
            Primitive::F32(f) => Primitive::F32(f32::from_ne_bytes(f.to_le_bytes())),
            Primitive::F64(f) => Primitive::F64(f64::from_ne_bytes(f.to_le_bytes())),
        }
    }
}

impl View for Primitive {
    fn dtype(&self) -> Dtype {
        match self {
            Primitive::Bool(_) => Dtype::BOOL,
            Primitive::I8(_) => Dtype::I8,
            Primitive::U8(_) => Dtype::U8,
            Primitive::I16(_) => Dtype::I16,
            Primitive::U16(_) => Dtype::U16,
            Primitive::I32(_) => Dtype::I32,
            Primitive::U32(_) => Dtype::U32,
            Primitive::I64(_) => Dtype::I64,
            Primitive::U64(_) => Dtype::U64,
            Primitive::F32(_) => Dtype::F32,
            Primitive::F64(_) => Dtype::F64,
        }
    }

    fn shape(&self) -> &[usize] {
        &[1]
    }

    fn data(&self) -> Cow<'_, [u8]> {
        match self {
            Primitive::Bool(b) => Cow::Borrowed(std::slice::from_ref(unsafe { std::mem::transmute::<_, &u8>(b) })),
            Primitive::I8(i) => Cow::Borrowed(std::slice::from_ref(unsafe { std::mem::transmute::<_, &u8>(i) })),
            Primitive::U8(u) => Cow::Borrowed(std::slice::from_ref(u)),
            Primitive::I16(i) => Cow::Borrowed(unsafe { std::mem::transmute::<_, &[u8; 2]>(i) }),
            Primitive::U16(u) => Cow::Borrowed(unsafe { std::mem::transmute::<_, &[u8; 2]>(u) }),
            Primitive::I32(i) => Cow::Borrowed(unsafe { std::mem::transmute::<_, &[u8; 4]>(i) }),
            Primitive::U32(u) => Cow::Borrowed(unsafe { std::mem::transmute::<_, &[u8; 4]>(u) }),
            Primitive::I64(i) => Cow::Borrowed(unsafe { std::mem::transmute::<_, &[u8; 8]>(i) }),
            Primitive::U64(u) => Cow::Borrowed(unsafe { std::mem::transmute::<_, &[u8; 8]>(u) }),
            Primitive::F32(f) => Cow::Borrowed(unsafe { std::mem::transmute::<_, &[u8; 4]>(f) }),
            Primitive::F64(f) => Cow::Borrowed(unsafe { std::mem::transmute::<_, &[u8; 8]>(f) }),
        }
    }

    fn data_len(&self) -> usize {
        self.data().len()
    }
}

macro_rules! from_primitive {
    ($ty:ty, $prim:ident) => {
        impl From<$ty> for Primitive {
            fn from(value: $ty) -> Self {
                Self::$prim(value)
            }
        }

        impl TryFrom<Primitive> for $ty {
            type Error = PrimitiveKind;
            fn try_from(value: Primitive) -> Result<Self, Self::Error> {
                if let Primitive::$prim(v) = value {
                    Ok(v)
                } else {
                    Err(value.kind())
                }
            }
        }
    };
}

from_primitive!(bool, Bool);
from_primitive!(i8, I8);
from_primitive!(u8, U8);
from_primitive!(i16, I16);
from_primitive!(u16, U16);
from_primitive!(i32, I32);
from_primitive!(u32, U32);
from_primitive!(i64, I64);
from_primitive!(u64, U64);
from_primitive!(f32, F32);
from_primitive!(f64, F64);

impl From<usize> for Primitive {
    fn from(value: usize) -> Self {
        Self::U64(value as _)
    }
}

impl TryFrom<Primitive> for usize {
    type Error = PrimitiveKind;
    fn try_from(value: Primitive) -> Result<Self, Self::Error> {
        if let Primitive::U64(v) = value {
            Ok(v as _)
        } else {
            Err(value.kind())
        }
    }
}

impl From<isize> for Primitive {
    fn from(value: isize) -> Self {
        Self::I64(value as _)
    }
}

impl TryFrom<Primitive> for isize {
    type Error = PrimitiveKind;
    fn try_from(value: Primitive) -> Result<Self, Self::Error> {
        if let Primitive::I64(v) = value {
            Ok(v as _)
        } else {
            Err(value.kind())
        }
    }
}
