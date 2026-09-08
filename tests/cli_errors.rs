//! CLI exit-code contract for the `forecast` binary.
//!
//! The `quantize` binary's CLI surface is covered in `quantize_presets.rs`
//! (which drives it via `CARGO_BIN_EXE_quantize`); here we lock the
//! `forecast` argument handling: missing args, unparsable horizon, and a
//! missing checkpoint directory must all fail fast with a nonzero status
//! instead of panicking or hanging.

use std::process::Command;

fn forecast_bin() -> &'static str {
    env!("CARGO_BIN_EXE_forecast")
}

#[test]
fn forecast_no_args_exits_2() {
    let out = Command::new(forecast_bin())
        .output()
        .expect("spawn forecast");
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("usage"), "must print usage, got: {err}");
}

#[test]
fn forecast_bad_horizon_fails() {
    let out = Command::new(forecast_bin())
        .arg("some_dir")
        .arg("not_a_number")
        .output()
        .expect("spawn forecast");
    assert!(!out.status.success(), "unparsable horizon must fail");
}

#[test]
fn forecast_missing_checkpoint_fails() {
    let out = Command::new(forecast_bin())
        .arg("definitely_not_a_checkpoint_dir")
        .arg("24")
        .output()
        .expect("spawn forecast");
    assert!(!out.status.success(), "missing checkpoint must fail");
    assert_eq!(out.status.code(), Some(2));
}
