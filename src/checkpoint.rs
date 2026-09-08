//! safetensors checkpoint loading.
//!
//! Format: 8-byte little-endian header length, then a JSON header mapping
//! tensor names to `{dtype, shape, data_offsets}`, then raw tensor bytes.
//! Tensor byte offsets are relative to the start of the data section.
//!
//! Supports zero-copy memory mapping (`mmap`) for checkpoints to eliminate load-time
//! memory duplication, as well as in-memory bytes for fixtures and embedding.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::json::Json;
use crate::tensor::Tensor2D;

/// safetensors dtype strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F64,
    F32,
    F16,
    I64,
    I32,
    I16,
    I8,
    U8,
    U32,
    U64,
    Bool,
}

impl DType {
    pub fn parse(s: &str) -> Option<DType> {
        Some(match s {
            "F64" => DType::F64,
            "F32" => DType::F32,
            "F16" => DType::F16,
            "I64" => DType::I64,
            "I32" => DType::I32,
            "I16" => DType::I16,
            "I8" => DType::I8,
            "U8" => DType::U8,
            "U32" => DType::U32,
            "U64" => DType::U64,
            "BOOL" => DType::Bool,
            _ => return None,
        })
    }
}

/// Metadata for one tensor in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte offset relative to the start of the data section.
    pub byte_offset: usize,
    pub byte_len: usize,
}

#[derive(Debug)]
enum DataStorage {
    Mmap(memmap2::Mmap),
    Vec(Vec<u8>),
}

impl std::ops::Deref for DataStorage {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            DataStorage::Mmap(m) => m.as_ref(),
            DataStorage::Vec(v) => v.as_slice(),
        }
    }
}

/// A parsed safetensors file.
#[derive(Debug)]
pub struct Safetensors {
    tensors: HashMap<String, TensorInfo>,
    storage: DataStorage,
    data_offset: usize,
}

impl Safetensors {
    /// Reads and parses a `.safetensors` file via memory mapping (zero copy).
    pub fn load(path: &Path) -> Result<Safetensors> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Safetensors::from_storage(DataStorage::Mmap(mmap))
    }

    /// Parses from in-memory bytes (used by tests and embedders).
    pub fn parse_bytes(bytes: &[u8]) -> Result<Safetensors> {
        Safetensors::from_storage(DataStorage::Vec(bytes.to_vec()))
    }

    /// Parses from owned in-memory bytes.
    pub fn from_owned_bytes(bytes: Vec<u8>) -> Result<Safetensors> {
        Safetensors::from_storage(DataStorage::Vec(bytes))
    }

    fn from_storage(storage: DataStorage) -> Result<Safetensors> {
        let bytes: &[u8] = &storage;
        if bytes.len() < 8 {
            return Err(Error(
                "safetensors: file shorter than 8-byte header length".into(),
            ));
        }
        let header_len = u64::from_le_bytes(
            bytes[..8]
                .try_into()
                .map_err(|_| Error("safetensors: header length truncated".into()))?,
        ) as usize;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| Error("safetensors: header length overflow".into()))?;
        if header_end > bytes.len() {
            return Err(Error(format!(
                "safetensors: header length {header_len} exceeds file size {}",
                bytes.len()
            )));
        }

        let root = Json::parse(&bytes[8..header_end])?;
        let obj = root
            .as_obj()
            .ok_or_else(|| Error("safetensors: header is not a JSON object".into()))?;

        let mut tensors = HashMap::new();
        for (name, val) in obj {
            if name == "__metadata__" {
                continue;
            }
            let meta = val.as_obj().ok_or_else(|| {
                Error(format!(
                    "safetensors: tensor '{name}' metadata is not an object"
                ))
            })?;
            let get = |key: &str| -> Result<&Json> {
                meta.iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v)
                    .ok_or_else(|| Error(format!("safetensors: tensor '{name}' missing '{key}'")))
            };

            let dtype_str = get("dtype")?
                .as_str()
                .ok_or_else(|| Error(format!("safetensors: tensor '{name}' dtype not a string")))?;
            let dtype = DType::parse(dtype_str).ok_or_else(|| {
                Error(format!(
                    "safetensors: tensor '{name}' unknown dtype '{dtype_str}'"
                ))
            })?;

            let shape = get("shape")?
                .as_arr()
                .ok_or_else(|| Error(format!("safetensors: tensor '{name}' shape not an array")))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| {
                            Error(format!(
                                "safetensors: tensor '{name}' shape has non-integer"
                            ))
                        })
                        .map(|d| d as usize)
                })
                .collect::<Result<Vec<usize>>>()?;

            let offs = get("data_offsets")?.as_arr().ok_or_else(|| {
                Error(format!(
                    "safetensors: tensor '{name}' data_offsets not an array"
                ))
            })?;
            if offs.len() != 2 {
                return Err(Error(format!(
                    "safetensors: tensor '{name}' data_offsets must have 2 entries"
                )));
            }
            let start = offs[0]
                .as_u64()
                .ok_or_else(|| Error(format!("safetensors: tensor '{name}' bad offset")))?
                as usize;
            let end = offs[1]
                .as_u64()
                .ok_or_else(|| Error(format!("safetensors: tensor '{name}' bad offset")))?
                as usize;
            if end < start {
                return Err(Error(format!(
                    "safetensors: tensor '{name}' offsets reversed"
                )));
            }

            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    byte_offset: start,
                    byte_len: end - start,
                },
            );
        }

        Ok(Safetensors {
            tensors,
            storage,
            data_offset: header_end,
        })
    }

    /// Number of tensors in the file.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Reads an F32 tensor as a rank-2 row-major `Tensor2D`.
    /// Reads a 2D tensor as a rank-2 row-major `Tensor2D` (auto-converts from F16 if needed).
    ///
    /// The checkpoint stores weights in torch layout `(out, in)`. This accessor
    /// returns them as-is; callers that need inference layout transpose once at
    /// load (see `model::residual_block::load_linear`).
    pub fn f32_tensor(&self, name: &str) -> Result<Tensor2D> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error(format!("safetensors: no tensor '{name}'")))?;
        if info.dtype != DType::F32 && info.dtype != DType::F16 {
            return Err(Error(format!(
                "safetensors: tensor '{name}' is {:?}, expected F32 or F16",
                info.dtype
            )));
        }
        if info.shape.len() != 2 {
            return Err(Error(format!(
                "safetensors: tensor '{name}' rank {} is not 2",
                info.shape.len()
            )));
        }
        let (rows, cols) = (info.shape[0], info.shape[1]);
        Tensor2D::from_row_major(rows, cols, self.f32_flat(name)?)
    }

    /// Reads a 2D tensor directly as row-major `half::f16` (torch layout
    /// `(out, in)`, unconverted). Errors when the stored dtype is not F16 —
    /// callers must match the checkpoint dtype instead of round-tripping
    /// through f32.
    pub fn f16_tensor(&self, name: &str) -> Result<(usize, usize, Vec<half::f16>)> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error(format!("safetensors: no tensor '{name}'")))?;
        if info.dtype != DType::F16 {
            return Err(Error(format!(
                "safetensors: tensor '{name}' is {:?}, expected F16",
                info.dtype
            )));
        }
        if info.shape.len() != 2 {
            return Err(Error(format!(
                "safetensors: tensor '{name}' rank {} is not 2",
                info.shape.len()
            )));
        }
        let (rows, cols) = (info.shape[0], info.shape[1]);
        Ok((rows, cols, self.f16_flat(name)?))
    }

    /// Reads a tensor as a flat `Vec<f32>` (rank-agnostic), auto-converting from F16 if needed.
    pub fn f32_flat(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error(format!("safetensors: no tensor '{name}'")))?;
        let data = &self.storage[self.data_offset..];
        let raw = data
            .get(info.byte_offset..info.byte_offset + info.byte_len)
            .ok_or_else(|| Error(format!("safetensors: tensor '{name}' data out of range")))?;
        match info.dtype {
            DType::F32 => {
                if raw.len() % 4 != 0 {
                    return Err(Error(format!(
                        "safetensors: tensor '{name}' byte length {} not a multiple of 4",
                        raw.len()
                    )));
                }
                let mut out = Vec::with_capacity(info.byte_len / 4);
                out.extend(
                    raw.as_chunks::<4>().0.iter()
                        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
                );
                Ok(out)
            }
            DType::F16 => {
                if raw.len() % 2 != 0 {
                    return Err(Error(format!(
                        "safetensors: tensor '{name}' byte length {} not a multiple of 2",
                        raw.len()
                    )));
                }
                let mut out = Vec::with_capacity(info.byte_len / 2);
                out.extend(
                    raw.as_chunks::<2>().0.iter()
                        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()),
                );
                Ok(out)
            }
            _ => Err(Error(format!(
                "safetensors: tensor '{name}' is {:?}, expected F32 or F16",
                info.dtype
            ))),
        }
    }

    /// Reads a tensor as flat `Vec<half::f16>`, auto-converting from F32 if needed.
    pub fn f16_flat(&self, name: &str) -> Result<Vec<half::f16>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error(format!("safetensors: no tensor '{name}'")))?;
        let data = &self.storage[self.data_offset..];
        let raw = data
            .get(info.byte_offset..info.byte_offset + info.byte_len)
            .ok_or_else(|| Error(format!("safetensors: tensor '{name}' data out of range")))?;
        match info.dtype {
            DType::F16 => {
                if raw.len() % 2 != 0 {
                    return Err(Error(format!(
                        "safetensors: tensor '{name}' byte length {} not a multiple of 2",
                        raw.len()
                    )));
                }
                let mut out = Vec::with_capacity(info.byte_len / 2);
                out.extend(
                    raw.as_chunks::<2>().0.iter()
                        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]]))),
                );
                Ok(out)
            }
            DType::F32 => {
                if raw.len() % 4 != 0 {
                    return Err(Error(format!(
                        "safetensors: tensor '{name}' byte length {} not a multiple of 4",
                        raw.len()
                    )));
                }
                let mut out = Vec::with_capacity(info.byte_len / 4);
                out.extend(raw.as_chunks::<4>().0.iter().map(|c| {
                    half::f16::from_f32(f32::from_bits(u32::from_le_bytes([
                        c[0], c[1], c[2], c[3],
                    ])))
                }));
                Ok(out)
            }
            _ => Err(Error(format!(
                "safetensors: tensor '{name}' is {:?}, expected F32 or F16",
                info.dtype
            ))),
        }
    }

    /// Returns an iterator over tensor names and their metadata.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.tensors.iter().map(|(k, v)| (k.as_str(), v))
    }
}

/// Borrowed tensor slice for serialization into safetensors format.
pub struct TensorView<'a> {
    pub name: &'a str,
    pub dtype: DType,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

/// Writes tensors into a valid `.safetensors` file.
pub fn write_safetensors(path: &Path, tensors: &[TensorView]) -> Result<()> {
    let mut header_map = Vec::new();
    let mut offset = 0usize;

    for t in tensors {
        let end = offset + t.data.len();
        let dtype_str = match t.dtype {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::Bool => "BOOL",
            _ => return Err(Error(format!("unsupported dtype {:?}", t.dtype))),
        };
        let shape_str = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        header_map.push(format!(
            "\"{}\": {{\"dtype\": \"{}\", \"shape\": [{}], \"data_offsets\": [{}, {}]}}",
            t.name, dtype_str, shape_str, offset, end
        ));
        offset = end;
    }

    let header_json = format!("{{{}}}", header_map.join(", "));
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;
    for t in tensors {
        file.write_all(t.data)?;
    }
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an in-memory safetensors blob with two F32 tensors.
    fn sample_blob() -> Vec<u8> {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b: Vec<f32> = vec![10.0, 20.0]; // [2]
        let a_off = 0usize;
        let a_end = a.len() * 4;
        let b_off = a_end;
        let b_end = b_off + b.len() * 4;
        let header = format!(
            "{{\"a\": {{\"dtype\": \"F32\", \"shape\": [2, 3], \"data_offsets\": [{a_off}, {a_end}]}}, \"b\": {{\"dtype\": \"F32\", \"shape\": [2], \"data_offsets\": [{b_off}, {b_end}]}}, \"__metadata__\": {{\"k\": \"v\"}}}}",
            a_off = a_off,
            a_end = a_end,
            b_off = b_off,
            b_end = b_end,
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        out.extend_from_slice(&b.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        out
    }

    #[test]
    fn parses_and_reads_tensors() {
        let st = Safetensors::parse_bytes(&sample_blob()).expect("test fixture invariant");
        assert_eq!(st.len(), 2);

        let a = st.f32_tensor("a").expect("test fixture invariant");
        assert_eq!((a.rows(), a.cols()), (2, 3));
        assert_eq!(a[(1, 2)], 6.0);

        let b = st.f32_flat("b").expect("test fixture invariant");
        assert_eq!(b, vec![10.0, 20.0]);
    }

    #[test]
    fn unknown_tensor_errors() {
        let st = Safetensors::parse_bytes(&sample_blob()).expect("test fixture invariant");
        assert!(st.f32_tensor("nope").is_err());
    }

    #[test]
    fn bad_length_errors() {
        assert!(Safetensors::parse_bytes(b"short").is_err());
        let mut blob = vec![0u8; 16];
        blob[..8].copy_from_slice(&1000u64.to_le_bytes());
        assert!(Safetensors::parse_bytes(&blob).is_err());
    }

    #[test]
    fn f16_roundtrip_test() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_f16_roundtrip.safetensors");

        let f16_vals: Vec<half::f16> = vec![
            half::f16::from_f32(1.5),
            half::f16::from_f32(2.5),
            half::f16::from_f32(-3.5),
            half::f16::from_f32(4.0),
        ];
        let f16_bytes: Vec<u8> = f16_vals
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let views = vec![TensorView {
            name: "weight_f16",
            dtype: DType::F16,
            shape: &[2, 2],
            data: &f16_bytes,
        }];

        write_safetensors(&path, &views).expect("write failed");
        let st = Safetensors::load(&path).expect("load failed");
        assert_eq!(st.len(), 1);

        // Read f16 as f32
        let f32_t = st.f32_tensor("weight_f16").expect("read f16 tensor");
        assert_eq!(f32_t[(0, 0)], 1.5);
        assert_eq!(f32_t[(0, 1)], 2.5);
        assert_eq!(f32_t[(1, 0)], -3.5);
        assert_eq!(f32_t[(1, 1)], 4.0);

        // Clean up
        let _ = std::fs::remove_file(path);
    }
}
