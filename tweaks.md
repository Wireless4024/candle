This crate only focused to run on nightly rust.

# Tensors
- [x] AVX512 support
- [x] u16 support (only cpu and cuda)
- [x] backward with detach option
- [x] more validation when construct tensor on mismatch input data and shape

# Vars
- [x] variable are replaced with named variable
- [x] add `GeometryAware` to var so it can use different optimizer (like Muon)

# Module
- [x] sequence use inline vtable instead of trait object
- [ ] parallel sequence dispatch

# misc
- [x] use ahash for hashmap
- [x] varbuilder use Arc&lt;str&gt; instead of String
- [x] cuda will preload all module and avoid mutex usage
- [x] serializable module
- [x] custom varmap (SafetensorStorage) that save metadata