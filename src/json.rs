//! Minimal JSON parser, scoped to safetensors header files.
//!
//! safetensors headers are small (KBs), deterministic JSON produced by the
//! huggingface ecosystem. Rather than pulling serde_json into the binary we
//! hand-roll a ~150-line recursive descent parser that covers everything a
//! header can legally contain. Unknown constructs surface as errors instead
//! of being silently skipped, so a format change fails loudly at load time.

use crate::error::{Error, Result};

/// Minimal JSON value tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(u64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Parses a complete JSON document from UTF-8 bytes.
    pub fn parse(bytes: &[u8]) -> Result<Json> {
        let mut p = Parser { b: bytes, i: 0 };
        let v = p.parse_value()?;
        p.skip_ws();
        if p.i != p.b.len() {
            return Err(Error(format!("json: trailing content at byte {}", p.i)));
        }
        Ok(v)
    }

    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(o) => Some(o),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Int(n) => Some(*n),
            _ => None,
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek();
        if c.is_some() {
            self.i += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_whitespace()) {
            self.i += 1;
        }
    }

    fn parse_value(&mut self) -> Result<Json> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => self.parse_string().map(Json::Str),
            Some(b't') => self.expect_literal(b"true").map(|_| Json::Bool(true)),
            Some(b'f') => self.expect_literal(b"false").map(|_| Json::Bool(false)),
            Some(b'n') => self.expect_literal(b"null").map(|_| Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(Error(format!("json: unexpected byte 0x{c:02x}"))),
            None => Err(Error("json: unexpected end of input".into())),
        }
    }

    fn parse_object(&mut self) -> Result<Json> {
        // NOTE: `self.bump()` must run unconditionally (it consumes '{');
        // putting it inside debug_assert_eq would strip the side effect in
        // release builds. This was a real released-path bug.
        let open = self.bump();
        debug_assert_eq!(open, Some(b'{'));
        let mut out: Vec<(String, Json)> = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(out));
                }
                Some(b'"') => {
                    let key = self.parse_string()?;
                    self.skip_ws();
                    if self.bump() != Some(b':') {
                        return Err(Error("json: expected ':' in object".into()));
                    }
                    let val = self.parse_value()?;
                    out.push((key, val));
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => {
                            self.i += 1;
                        }
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(out));
                        }
                        _ => return Err(Error("json: expected ',' or '}' in object".into())),
                    }
                }
                _ => return Err(Error("json: expected object key or '}'".into())),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Json> {
        // See parse_object: bump must run in release builds too.
        let open = self.bump();
        debug_assert_eq!(open, Some(b'['));
        let mut out: Vec<Json> = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(out));
                }
                _ => {
                    out.push(self.parse_value()?);
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => {
                            self.i += 1;
                        }
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(out));
                        }
                        _ => return Err(Error("json: expected ',' or ']' in array".into())),
                    }
                }
            }
        }
    }

    /// Parses a JSON string (opening quote not yet consumed) into UTF-8 bytes.
    fn parse_string(&mut self) -> Result<String> {
        debug_assert_eq!(self.peek(), Some(b'"'));
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            match self.bump() {
                Some(b'"') => {
                    return String::from_utf8(out)
                        .map_err(|e| Error(format!("json: invalid utf-8 in string: {e}")));
                }
                Some(b'\\') => match self.bump() {
                    Some(b'"') => out.push(b'"'),
                    Some(b'\\') => out.push(b'\\'),
                    Some(b'/') => out.push(b'/'),
                    Some(b'b') => out.push(0x08),
                    Some(b'f') => out.push(0x0c),
                    Some(b'n') => out.push(b'\n'),
                    Some(b'r') => out.push(b'\r'),
                    Some(b't') => out.push(b'\t'),
                    Some(b'u') => {
                        let mut cp: u32 = 0;
                        for _ in 0..4 {
                            let h = self
                                .bump()
                                .ok_or_else(|| Error("json: truncated \\u escape".into()))?;
                            let d = (h as char).to_digit(16).ok_or_else(|| {
                                Error(format!("json: bad hex digit in \\u escape: 0x{h:02x}"))
                            })?;
                            cp = cp * 16 + d;
                        }
                        // Surrogate pairs are rejected; safetensors headers are ASCII.
                        let ch = char::from_u32(cp).ok_or_else(|| {
                            Error("json: invalid code point in \\u escape".into())
                        })?;
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                    Some(c) => return Err(Error(format!("json: bad escape: 0x{c:02x}"))),
                    None => return Err(Error("json: unterminated string".into())),
                },
                Some(c) => out.push(c),
                None => return Err(Error("json: unterminated string".into())),
            }
        }
    }

    fn parse_number(&mut self) -> Result<Json> {
        let start = self.i;
        let mut is_float = false;
        while let Some(c) = self.peek() {
            match c {
                b'-' => self.i += 1, // leading sign (u64/i64 parse handles it)
                b'0'..=b'9' => self.i += 1,
                b'.' => {
                    is_float = true;
                    self.i += 1;
                    // Digits after '.' are consumed by the digit arm.
                }
                b'e' | b'E' => {
                    is_float = true;
                    self.i += 1;
                    if self.peek().is_some_and(|c| c == b'+' || c == b'-') {
                        self.i += 1;
                    }
                }
                _ => break,
            }
        }
        let raw = &self.b[start..self.i];
        if raw.is_empty() || raw == b"-" {
            return Err(Error("json: malformed number".into()));
        }
        let s =
            std::str::from_utf8(raw).map_err(|e| Error(format!("json: number not utf-8: {e}")))?;
        if is_float {
            let v: f64 = s
                .parse()
                .map_err(|e| Error(format!("json: bad float: {e}")))?;
            Ok(Json::Float(v))
        } else {
            // Pure integer (may be negative): try u64 first, then i64.
            if let Ok(v) = s.parse::<u64>() {
                return Ok(Json::Int(v));
            }
            if let Ok(v) = s.parse::<i64>() {
                // Keep the enum minimal: store signed values as Float.
                return Ok(Json::Float(v as f64));
            }
            Err(Error(format!("json: bad integer: {s}")))
        }
    }

    fn expect_literal(&mut self, lit: &[u8]) -> Result<()> {
        if self.b.len() - self.i >= lit.len() && &self.b[self.i..self.i + lit.len()] == lit {
            self.i += lit.len();
            Ok(())
        } else {
            Err(Error(format!(
                "json: expected literal '{}'",
                String::from_utf8_lossy(lit)
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj_field<'a>(obj: &'a [(String, Json)], k: &str) -> Option<&'a Json> {
        obj.iter().find(|(name, _)| name == k).map(|(_, v)| v)
    }

    #[test]
    fn parses_safetensors_shaped_header() {
        let src = br#"{"tensor": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}, "__metadata__": {"n": 1}}"#;
        let j = Json::parse(src).expect("test fixture invariant");
        let obj = j.as_obj().expect("test fixture invariant");
        let t = obj_field(obj, "tensor")
            .expect("test fixture invariant")
            .as_obj()
            .expect("test fixture invariant");
        assert_eq!(
            obj_field(t, "dtype")
                .expect("test fixture invariant")
                .as_str(),
            Some("F32")
        );
        let shape = obj_field(t, "shape")
            .expect("test fixture invariant")
            .as_arr()
            .expect("test fixture invariant");
        assert_eq!(shape[0].as_u64(), Some(2));
        assert_eq!(shape[1].as_u64(), Some(3));
        let offs = obj_field(t, "data_offsets")
            .expect("test fixture invariant")
            .as_arr()
            .expect("test fixture invariant");
        assert_eq!(offs[1].as_u64(), Some(24));
    }

    #[test]
    fn string_escapes() {
        let j = Json::parse(br#""a\"b\\c\/d\n\u0041""#).expect("test fixture invariant");
        assert_eq!(j.as_str(), Some("a\"b\\c/d\nA"));
    }

    #[test]
    fn rejects_trailing_garbage() {
        assert!(Json::parse(b"{} x").is_err());
        assert!(Json::parse(b"").is_err());
        assert!(Json::parse(b"123").is_ok());
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;

    #[test]
    fn booleans_and_floats() {
        let cases: &[&[u8]] = &[
            br#"{"a": false, "b": true}"#.as_slice(),
            br#"{"a": 1e+20}"#.as_slice(),
            br#"{"a": "x", "b": 1.5}"#.as_slice(),
            br#"{"a": [0.1, 0.5, 0.9]}"#.as_slice(),
            br#"{"n": null, "x": -3}"#.as_slice(),
        ];
        for src in cases {
            let ok = Json::parse(src).is_ok();
            println!("{} → ok={}", String::from_utf8_lossy(src), ok);
            assert!(ok, "should parse: {}", String::from_utf8_lossy(src));
        }
    }
}

#[cfg(test)]
mod file_tests {
    use super::*;

    #[test]
    fn parses_actual_golden_config() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("golden/e2e_micro/config.json");
        let bytes = std::fs::read(&path).expect("read config.json");
        match Json::parse(&bytes) {
            Ok(j) => println!("parsed OK: {:?}", j.as_obj().map(|o| o.len())),
            Err(e) => panic!("failed: {e}"),
        }
    }
}

#[cfg(test)]
mod real_config_tests {
    use super::*;

    #[test]
    fn parses_real_timesfm_config() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ckpt/config.json");
        if !path.exists() {
            eprintln!("ckpt/config.json not present — skip");
            return;
        }
        let bytes = std::fs::read(&path).expect("read");
        match Json::parse(&bytes) {
            Ok(j) => {
                let obj = j.as_obj().expect("obj");
                println!("OK: {} top-level keys", obj.len());
            }
            Err(e) => panic!("real config parse failed: {e}"),
        }
    }
}

#[cfg(test)]
mod sanity {
    use super::*;
    #[test]
    fn single_key_object() {
        assert!(Json::parse(br#"{"input_patch_len": 32}"#).is_ok());
        assert!(Json::parse(br#"{"a": 1, "b": 2}"#).is_ok());
    }
}
