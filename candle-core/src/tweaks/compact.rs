use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    alloc::{alloc, handle_alloc_error, Layout},
    borrow::{Borrow, Cow},
    hash::Hash,
    io::Write,
    ops::Deref,
    ptr::slice_from_raw_parts_mut,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

const INNER_SIZE: usize = size_of::<usize>();

pub struct ArcStr {
    //inner: *mut ArcInner,
    inner: Option<Arc<str>>,
}

impl Default for ArcStr {
    fn default() -> Self {
        Self::empty()
    }
}

impl Serialize for ArcStr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_str().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ArcStr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let str = crate::tweaks::serde::borrow_cow_str::<_, Cow<str>>(deserializer)?;
        Ok(ArcStr::new_copy(&str))
    }
}

impl Eq for ArcStr {}
impl PartialEq for ArcStr {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl PartialOrd for ArcStr {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ArcStr {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}
impl Hash for ArcStr {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

pub trait ToArcString {
    fn to_string(&self) -> ArcStr;
}

impl ToArcString for ArcStr {
    #[inline(always)]
    fn to_string(&self) -> ArcStr {
        self.clone()
    }
}

impl ToArcString for &ArcStr {
    #[inline(always)]
    fn to_string(&self) -> ArcStr {
        (*self).clone()
    }
}

impl ToArcString for str {
    #[inline(always)]
    fn to_string(&self) -> ArcStr {
        ArcStr::new_copy(self)
    }
}

impl ToArcString for &str {
    #[inline(always)]
    fn to_string(&self) -> ArcStr {
        ArcStr::new_copy(self)
    }
}

impl ToArcString for String {
    #[inline(always)]
    fn to_string(&self) -> ArcStr {
        ArcStr::new_copy(self)
    }
}
impl ToArcString for Box<str> {
    #[inline(always)]
    fn to_string(&self) -> ArcStr {
       ArcStr::new_copy(self)
    }
}

impl ToArcString for Arc<str> {
    #[inline(always)]
    fn to_string(&self) -> ArcStr {
        ArcStr { inner: Some(self.clone()) }
    }
}

impl ToArcString for &String {
    #[inline(always)]
    fn to_string(&self) -> ArcStr {
        ArcStr::new_copy(self)
    }
}
macro_rules! forward_to_str {
    ($ty:ty) => {
        impl From<$ty> for ArcStr {
            #[inline(always)]
            fn from(value: $ty) -> Self {
                ToArcString::to_string(&value)
            }
        }
    };
}
forward_to_str!(&ArcStr);
macro_rules! numeric_to_str {
    ($ty:ty) => {
        impl ToArcString for $ty {
            #[inline(always)]
            fn to_string(&self) -> ArcStr {
                ArcStr::new(self)
            }
        }
        forward_to_str!($ty);
    };
}
numeric_to_str!(u8);
numeric_to_str!(u16);
numeric_to_str!(u32);
numeric_to_str!(u64);
numeric_to_str!(u128);
numeric_to_str!(usize);
numeric_to_str!(i8);
numeric_to_str!(i16);
numeric_to_str!(i32);
numeric_to_str!(i64);
numeric_to_str!(i128);
numeric_to_str!(isize);

impl From<&str> for ArcStr {
    #[inline(always)]
    fn from(value: &str) -> Self {
        ArcStr::new_copy(value)
    }
}
impl From<String> for ArcStr {
    #[inline(always)]
    fn from(value: String) -> Self {
        ArcStr::new_copy(&value)
    }
}
impl From<&String> for ArcStr {
    #[inline(always)]
    fn from(value: &String) -> Self {
        ArcStr::new_copy(value)
    }
}
impl Borrow<str> for ArcStr {
    #[inline(always)]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}
impl std::fmt::Display for ArcStr {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

impl std::fmt::Debug for ArcStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

unsafe impl Send for ArcStr {}
unsafe impl Sync for ArcStr {}

impl AsRef<str> for ArcStr {
    #[inline(always)]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[allow(dead_code)]
impl ArcStr {
    #[inline(always)]
    pub fn new<S: std::fmt::Display>(value: S) -> Self {
        let mut writer = StrWriter::new(32);
        write!(writer, "{value}").unwrap();
        writer.into_arc_str()
        //ArcStr::new_copy(&format!("{}", value))
    }

    #[inline(never)]
    fn allocate(len: usize) -> *mut ArcInner {
        unsafe {
            let layout = Layout::from_size_align_unchecked(INNER_SIZE + len, INNER_SIZE).pad_to_align();
            let data = alloc(layout);
            if data.is_null() {
                handle_alloc_error(layout);
            }
            let data = slice_from_raw_parts_mut(data, len) as *mut ArcInner;
            (*data).strong.store(1, Ordering::Relaxed);
            data
        }
    }

    pub const fn empty() -> Self {
        // ArcStr {
        //     inner: slice_from_raw_parts_mut(std::ptr::null_mut::<u8>(), 0) as *mut ArcInner,
        // }
        Self{
            inner: None,
        }
    }

    #[inline(always)]
    fn new_copy(str: &str) -> Self {
        if str.is_empty() {
            return Self::empty();
        }
        // unsafe {
        //     let data = Self::allocate(str.len());
        //     (*data).data.copy_from_slice(str.as_bytes());
        //     ArcStr { inner: data }
        // }

        ArcStr { inner: Some(str.into()) }
    }

    pub fn join<I: IntoIterator<Item = S>, S: AsRef<str>>(sep: &str, list: I) -> ArcStr
    where
        I::IntoIter: Clone,
    {
        let list = list.into_iter();
        // let count = list.clone().count();
        // if count == 0 {
        //     return Self::empty();
        // }
        // if count == 1 {
        //     return ArcStr::new_copy(list.next().unwrap().as_ref());
        // }
        // let len = list.clone().map(|s| s.as_ref().len()).sum::<usize>() + ((count - 1) * sep.len());
        // let this = Self::allocate(len);
        // unsafe {
        //     let mut ptr = this.as_mut_unchecked().data.as_mut_ptr();
        //     let end = ptr.add(len);
        //     for (i, s) in list.enumerate() {
        //         let s = s.as_ref();
        //         ptr.copy_from_nonoverlapping(s.as_bytes().as_ptr(), s.len());
        //         ptr = ptr.add(s.len());
        //         if i != count - 1 {
        //             ptr.copy_from_nonoverlapping(sep.as_bytes().as_ptr(), sep.len());
        //             ptr = ptr.add(sep.len());
        //         }
        //         debug_assert!(ptr <= end);
        //     }
        //     debug_assert!(ptr == end);
        //     ArcStr { inner: this }
        // }

        // TODO: optimize & fix buffer overflow
        let mut s = String::new();
        let count = list.clone().count();
        for (i, item) in list.enumerate() {
            s.push_str(item.as_ref());
            if i != count - 1 {
                s.push_str(sep);
            }
        }
        ArcStr::new_copy(&s)
    }

    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        //unsafe {
            self.inner.as_ref().map(|inner| inner.as_bytes()).unwrap_or(&[])
        //}
    }

    #[inline(always)]
    pub fn as_str(&self) -> &str {
        self.inner.as_deref().unwrap_or("")
    }

    pub fn cvt_string(&self) -> String {
        String::from(self.as_str())
    }
}

impl Deref for ArcStr {
    type Target = str;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl Clone for ArcStr {
    #[inline(always)]
    fn clone(&self) -> Self {
        // unsafe {
        //     if let Some(inner) = self.inner.as_ref() {
        //         inner.strong.fetch_add(1, Ordering::Relaxed);
        //         ArcStr { inner: self.inner }
        //     } else {
        //         ArcStr::empty()
        //     }
        // }
        ArcStr { inner: self.inner.clone() }
    }
}
// impl Drop for ArcStr {
//     #[inline(always)]
//     fn drop(&mut self) {
//         let Some(inner) = (unsafe { self.inner.as_ref() }) else {
//             return;
//         };
//         let strong = inner.strong.fetch_sub(1, Ordering::Relaxed);
//         if strong == 1 {
//             #[inline(never)]
//             fn drop_inner(inner: *mut ArcInner) {
//                 unsafe {
//                     let layout =
//                         Layout::from_size_align_unchecked(INNER_SIZE + inner.as_ref_unchecked().data.len(), INNER_SIZE)
//                             .pad_to_align();
//                     dealloc(inner.cast(), layout);
//                 }
//             }
//             drop_inner(self.inner);
//         }
//     }
// }

impl From<ArcStr> for String {
    #[inline(always)]
    fn from(value: ArcStr) -> Self {
        String::from(value.as_str())
    }
}

#[repr(C, align(8))]
pub struct ArcInner {
    strong: AtomicUsize,
    data: [u8],
}

// TODO: check if this is safe and alloc is valid
// struct StrWriter {
//     inner: *mut u8,
//     len: usize,
//     cap: usize,
// }
struct StrWriter {
    inner: String
}

#[allow(dead_code)]
impl StrWriter {
    fn new(cap:usize)->Self{
        Self{
            inner: String::with_capacity(cap),
        }
    }
    // fn new(cap: usize) -> Self {
    //     let layout = Self::layout(cap);
    //     let data = unsafe { alloc(layout) };
    //     if data.is_null() {
    //         handle_alloc_error(layout);
    //     }
    //     Self {
    //         inner: data,
    //         len: 0,
    //         cap,
    //     }
    // }
    #[inline(always)]
    fn layout(cap: usize) -> Layout {
        unsafe { Layout::from_size_align_unchecked(INNER_SIZE + cap, INNER_SIZE).pad_to_align() }
    }
    #[inline(never)]
    fn grow(&mut self, _cap: usize) {
        // let layout = Self::layout(self.cap);
        // let new_layout = Self::layout(cap);
        // let data = unsafe { realloc(self.inner, layout, new_layout.size()) };
        // if data.is_null() {
        //     handle_alloc_error(layout);
        // }
        // self.inner = data;
        // self.cap = new_layout.size();
    }
    #[inline]
    fn into_arc_str(self) -> ArcStr {
        // if self.len == 0 {
        //     unsafe {
        //         dealloc(self.inner, Self::layout(self.cap));
        //     }
        //     return ArcStr::empty();
        // }
        // self.grow(self.len + INNER_SIZE);
        // let inner = slice_from_raw_parts_mut(self.inner, self.len) as *mut ArcInner;
        // ArcStr { inner }
        ArcStr { inner: Some(self.inner.as_str().into()) }
    }
}

impl Write for StrWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // let len = buf.len();
        // let remain = self.cap - self.len;
        // unlikely(len > remain);
        // if len > remain {
        //     self.grow((self.cap + len).next_power_of_two());
        // }
        // unsafe {
        //     std::ptr::copy_nonoverlapping(buf.as_ptr(), self.inner.add(INNER_SIZE + self.len), len);
        // }
        // self.len += len;
        use std::fmt::Write as _;
        self.inner.write_str(std::str::from_utf8(buf).unwrap()).unwrap();
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_str() {
        let s = ArcStr::new("test");
        assert_eq!(s.as_str(), "test");
    }
    #[test]
    fn test_arc_str_join() {
        let s = ArcStr::join(".", ["hello", "world"]);
        assert_eq!(s.as_str(), "hello.world");
    }
}
