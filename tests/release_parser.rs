//! Regression test for the release-only JSON parser bug.
//!
//! `parse_object`/`parse_array` consumed their open bracket *inside*
//! `debug_assert_eq!(self.bump(), ...)` — a side-effect that release builds
//! strip entirely, so `{`/`[` were never advanced past. Debug (`cargo test`)
//! masked it; this test is run as a normal integration test (and also in
//! `cargo test --release`) to keep the released path honest.

use timesfm3::json::Json;

macro_rules! assert_ok {
    ($src:expr) => {
        match Json::parse($src.as_bytes()) {
            Ok(_) => {}
            Err(e) => panic!("expected OK for {:?}, got {e}", $src),
        }
    };
}

#[test]
fn objects_and_arrays_parse_in_release() {
    for src in [
        "{}",
        "{\"a\":1}",
        "{\"a\": 1}",
        "{\"a\":1,\"b\":2}",
        "{\"input_patch_len\": 32}",
        "{\"a\":[1,2],\"b\":{\"c\":true}}",
        "[1,2]",
        "[]",
        "[[1],[2,3]]",
        "[{\"x\":1},{\"y\":2}]",
        "\"abc\"",
        "123",
        "-3",
        "1.5",
        "null",
        "true",
        "false",
    ] {
        assert_ok!(src);
    }
}

#[test]
fn nested_values_are_correct() {
    // Verify values survive the parse (not just "parses without error").
    let root = Json::parse(br#"{"a":[1,2.5],"b":{"c":true,"d":"x"}}"#).unwrap();
    let obj = root.as_obj().unwrap();
    let a = &obj.iter().find(|(k, _)| k == "a").unwrap().1;
    let arr = a.as_arr().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0].as_u64(), Some(1));
    let b = &obj.iter().find(|(k, _)| k == "b").unwrap().1;
    let bobj = b.as_obj().unwrap();
    let cf = &bobj.iter().find(|(k, _)| k == "c").unwrap().1;
    assert!(matches!(cf, Json::Bool(true)));
}
