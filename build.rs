use std::{env, ffi::OsString, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/ane/native.m");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        Ok("x86_64") => "x86_64",
        other => panic!("unsupported macOS native architecture: {other:?}"),
    };
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let object = out.join("dyn_ane_native.o");
    let library = out.join("libdyn_ane_native.a");
    let mut compiler = Command::new("xcrun");
    compiler.args([
        "--sdk",
        "macosx",
        "clang",
        "-arch",
        arch,
        "-fobjc-arc",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-c",
    ]);
    if let Some(target) = env::var_os("MACOSX_DEPLOYMENT_TARGET") {
        let mut flag = OsString::from("-mmacosx-version-min=");
        flag.push(target);
        compiler.arg(flag);
    }
    let status = compiler
        .arg("src/ane/native.m")
        .arg("-o")
        .arg(&object)
        .status()
        .expect("launch Objective-C compiler");
    assert!(status.success(), "Objective-C compilation failed");
    let status = Command::new("ar")
        .arg("crs")
        .arg(&library)
        .arg(&object)
        .status()
        .expect("launch archiver");
    assert!(status.success(), "native archive failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=dyn_ane_native");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=IOSurface");
}
