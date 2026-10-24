use super::*;

#[test]
fn startup_fallback_requires_completed_cleanup_through_error_contexts() {
    let clean = anyhow::Error::new(Error::new("compile unavailable".into()))
        .context("prepare ANE before serving");
    assert!(startup_fallback_allowed(&clean));
    let retained = anyhow::Error::new(
        Error::new("load failed and unload did not complete".into()).without_completed_cleanup(),
    )
    .context("prepare ANE before serving")
    .context("startup failed");
    assert!(!startup_fallback_allowed(&retained));
}

#[test]
fn projection_remains_neither_send_nor_sync() {
    // Each inferred marker has one implementation only while the owner lacks
    // the corresponding auto trait. Adding that trait makes inference ambiguous.
    trait AmbiguousIfSend<A> {
        fn marker() {}
    }
    impl<T: ?Sized> AmbiguousIfSend<()> for T {}
    impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}
    trait AmbiguousIfSync<A> {
        fn marker() {}
    }
    impl<T: ?Sized> AmbiguousIfSync<()> for T {}
    impl<T: ?Sized + Sync> AmbiguousIfSync<u8> for T {}
    let _ = <DynamicProjection as AmbiguousIfSend<_>>::marker;
    let _ = <DynamicProjection as AmbiguousIfSync<_>>::marker;
}

#[test]
fn exact_gate_layout_and_mil_have_no_constant_weight_route() {
    let shape = ProjectionShape::new(4096, 2560, 6144).unwrap();
    let layout = DynamicLayout::new(shape).unwrap();
    assert_eq!(layout.spatial, 10240);
    assert_eq!(layout.input_elements * 2, 50 * 1024 * 1024);
    assert_eq!(shape.output_elements().unwrap() * 2, 48 * 1024 * 1024);
    let source = mil(shape).unwrap();
    assert!(source.contains("tensor<fp16, [1, 2560, 1, 10240]> x"));
    assert!(source.contains("tensor<fp16, [1,1,4096,6144]> yh = matmul"));
    assert!(source.contains("tensor<fp16, [1,6144,1,4096]> y"));
    assert!(!source.contains("BLOBFILE"));
    assert!(!source.contains("conv("));
    assert!(source.starts_with(
        "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\""
    ));
}

#[test]
fn native_creation_mil_has_fresh_names_without_graph_changes() {
    unsafe extern "C" {
        fn dyn_ane_dense_test_copy_creation_mil(
            mil: *const u8,
            length: usize,
            output: *mut u8,
            capacity: usize,
            written: *mut usize,
        ) -> c_int;
    }
    let source = mil(ProjectionShape::new(64, 512, 512).unwrap()).unwrap();
    assert_eq!(source.matches("name=string(\"mm\")").count(), 1);
    let copy = || {
        let mut output = vec![0_u8; source.len() + 64];
        let mut written = 0;
        assert_eq!(
            unsafe {
                dyn_ane_dense_test_copy_creation_mil(
                    source.as_ptr(),
                    source.len(),
                    output.as_mut_ptr(),
                    output.len(),
                    &mut written,
                )
            },
            0
        );
        assert!(written <= output.len());
        output.truncate(written);
        output
    };
    let first = copy();
    let second = copy();
    assert!(
        first != second,
        "two native model creations reused identical MIL identity"
    );
    for bytes in [first, second] {
        let actual = std::str::from_utf8(&bytes).unwrap();
        let prefix = "name=string(\"mm_";
        assert_eq!(actual.matches(prefix).count(), 1);
        let (before, named) = actual.split_once(prefix).unwrap();
        let (nonce, after) = named.split_once("\")").unwrap();
        assert_eq!(nonce.len(), 32);
        assert!(nonce.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(format!("{before}name=string(\"mm\"){after}"), source);
    }
}

#[test]
fn native_creation_mil_rejects_missing_or_duplicate_names_without_output() {
    unsafe extern "C" {
        fn dyn_ane_dense_test_copy_creation_mil(
            mil: *const u8,
            length: usize,
            output: *mut u8,
            capacity: usize,
            written: *mut usize,
        ) -> c_int;
    }
    let source = mil(ProjectionShape::new(64, 512, 512).unwrap()).unwrap();
    for malformed in [
        String::new(),
        source.replace("name=string(\"mm\")", "name=string(\"other\")"),
        format!("{source}\nname=string(\"mm\")"),
    ] {
        let mut output = vec![0xa5_u8; malformed.len() + 64];
        let mut written = 0;
        assert_eq!(
            unsafe {
                dyn_ane_dense_test_copy_creation_mil(
                    malformed.as_ptr(),
                    malformed.len(),
                    output.as_mut_ptr(),
                    output.len(),
                    &mut written,
                )
            },
            -1
        );
        assert_eq!(written, 0);
        assert!(output.iter().all(|&byte| byte == 0xa5));
    }
}

#[test]
fn shape_rejects_invalid_dimensions_and_combined_extent_overflow() {
    for shape in [
        ProjectionShape {
            rows: 0,
            input_channels: 64,
            output_channels: 64,
        },
        ProjectionShape {
            rows: 65,
            input_channels: 64,
            output_channels: 64,
        },
        ProjectionShape {
            rows: 64,
            input_channels: 0,
            output_channels: 64,
        },
        ProjectionShape {
            rows: 64,
            input_channels: 64,
            output_channels: 0,
        },
        ProjectionShape {
            rows: 64,
            input_channels: usize::MAX,
            output_channels: 64,
        },
        ProjectionShape {
            rows: 2_147_483_632,
            input_channels: 1,
            output_channels: 64,
        },
    ] {
        assert!(DynamicLayout::new(shape).is_err(), "accepted {shape:?}");
    }
    assert!(checked_product(usize::MAX, 2).is_err());
    assert!(checked_product(isize::MAX as usize / 2 + 1, 1).is_err());
}

#[test]
fn readiness_requires_input_then_completed_evaluation() {
    let mut state = ExecutionState::default();
    assert!(state.begin_evaluate().is_err());
    assert!(state.require_output().is_err());
    state.begin_write().unwrap();
    state.input_written();
    assert!(state.require_output().is_err());
    state.begin_evaluate().unwrap();
    assert!(state.require_output().is_err());
    state.output_completed();
    state.require_output().unwrap();
    state.require_output().unwrap();
    state.begin_evaluate().unwrap();
    assert!(state.require_output().is_err());
    state.output_completed();
    state.begin_write().unwrap();
    assert!(state.begin_evaluate().is_err());
    assert!(state.require_output().is_err());
}

#[test]
fn every_native_operation_failure_poison_is_sticky_and_clears_readiness() {
    for stage in ["write", "evaluate", "read"] {
        let mut state = ExecutionState::default();
        state.input_written();
        state.output_completed();
        let mut error = [0; 2048];
        for (destination, byte) in error.iter_mut().zip(stage.as_bytes()) {
            *destination = *byte as c_char;
        }
        assert_eq!(
            state.finish_native(-1, &error).unwrap_err().to_string(),
            stage
        );
        assert!(!state.input_ready);
        assert!(!state.output_ready);
        assert!(state.healthy().is_err());
        assert!(state.begin_write().is_err());
        assert!(state.begin_evaluate().is_err());
        assert!(state.require_output().is_err());
        assert!(state.finish_native(0, &error).is_err());
    }
}

#[test]
fn wrong_host_length_is_rejected_before_native_execution() {
    let layout = DynamicLayout::new(ProjectionShape::new(64, 3, 2).unwrap()).unwrap();
    assert_eq!(layout.spatial, 66);
    assert_eq!(layout.input_elements, 198);
    check_length(198, layout.input_elements, "combined input").unwrap();
    for count in [0, 197, 199] {
        assert!(check_length(count, layout.input_elements, "combined input").is_err());
    }
    check_length(128, layout.shape.output_elements().unwrap(), "output").unwrap();
    assert!(check_length(127, layout.shape.output_elements().unwrap(), "output").is_err());
}

#[test]
fn native_pair_reference_survives_failed_create_and_unload() {
    unsafe extern "C" {
        fn dyn_ane_dense_test_shared_io_lifetime() -> c_int;
    }
    // Empty native allocations only: no IOSurfaces, ANE classes or selectors.
    assert_eq!(unsafe { dyn_ane_dense_test_shared_io_lifetime() }, 0);
}

#[test]
fn exclusive_claim_failure_cleanup_preserves_occupied_paths() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{MetadataExt, symlink};
    unsafe extern "C" {
        fn dyn_ane_dense_test_claim_and_cleanup(
            path: *const c_char,
            error: *mut c_char,
            error_cap: usize,
            cleanup_complete: *mut c_int,
        ) -> c_int;
    }
    let scratch = std::env::temp_dir().join(format!(
        "ane-exclusive-claim-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&scratch).unwrap();
    let directory = scratch.join("directory");
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("preserve"), b"occupied directory").unwrap();
    let file = scratch.join("file");
    std::fs::write(&file, b"occupied file").unwrap();
    let link = scratch.join("symlink");
    symlink(&directory, &link).unwrap();
    let claim = |path: &std::path::Path| {
        let path = CString::new(path.as_os_str().as_bytes()).unwrap();
        let mut error = [0; 2048];
        let mut cleanup_complete = 0;
        let status = unsafe {
            dyn_ane_dense_test_claim_and_cleanup(
                path.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
                &mut cleanup_complete,
            )
        };
        (status, cleanup_complete, native_error(&error).to_string())
    };
    for path in [&directory, &file, &link] {
        let before = std::fs::symlink_metadata(path).unwrap();
        let (status, cleanup_complete, error) = claim(path);
        assert_eq!(status, -1);
        assert_eq!(cleanup_complete, 1);
        assert!(error.contains("temporary directory was not created exclusively"));
        let after = std::fs::symlink_metadata(path).unwrap();
        assert_eq!((after.dev(), after.ino()), (before.dev(), before.ino()));
        assert_eq!(after.file_type(), before.file_type());
        assert_eq!(after.mode(), before.mode());
        assert_eq!(
            std::fs::read(directory.join("preserve")).unwrap(),
            b"occupied directory"
        );
        assert_eq!(std::fs::read(&file).unwrap(), b"occupied file");
        assert_eq!(std::fs::read_link(&link).unwrap(), directory);
    }
    let fresh = scratch.join("fresh");
    let (status, cleanup_complete, _) = claim(&fresh);
    assert_eq!(status, 0);
    assert_eq!(cleanup_complete, 1);
    assert!(!fresh.exists());
    std::fs::remove_dir_all(&scratch).unwrap();
}

#[test]
fn owned_cleanup_accepts_absence_but_preserves_replacement_and_symlink() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{MetadataExt, symlink};
    unsafe extern "C" {
        fn dyn_ane_dense_remove_owned_directory(
            path: *const c_char,
            device: u64,
            inode: u64,
            error: *mut c_char,
            error_cap: usize,
        ) -> c_int;
    }
    let scratch = std::env::temp_dir().join(format!(
        "ane-dynamic-cleanup-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&scratch).unwrap();
    let directory = scratch.join("owned");
    std::fs::create_dir(&directory).unwrap();
    let metadata = std::fs::metadata(&directory).unwrap();
    let path = CString::new(directory.as_os_str().as_bytes()).unwrap();
    let remove = |error: &mut [c_char; 2048]| unsafe {
        dyn_ane_dense_remove_owned_directory(
            path.as_ptr(),
            metadata.dev(),
            metadata.ino(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    let mut error = [0; 2048];
    std::fs::write(directory.join("payload"), b"owned").unwrap();
    assert_eq!(remove(&mut error), 0);
    assert_eq!(remove(&mut error), 0);
    std::fs::create_dir(&directory).unwrap();
    let replacement = std::fs::metadata(&directory).unwrap();
    let held = scratch.join("held");
    std::fs::rename(&directory, &held).unwrap();
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("preserve"), b"unowned").unwrap();
    assert_ne!(
        unsafe {
            dyn_ane_dense_remove_owned_directory(
                path.as_ptr(),
                replacement.dev(),
                replacement.ino(),
                error.as_mut_ptr(),
                error.len(),
            )
        },
        0
    );
    assert!(
        native_error(&error)
            .to_string()
            .contains("identity changed")
    );
    assert_eq!(
        std::fs::read(directory.join("preserve")).unwrap(),
        b"unowned"
    );
    std::fs::rename(&directory, scratch.join("replacement")).unwrap();
    symlink(&held, &directory).unwrap();
    assert_ne!(remove(&mut error), 0);
    assert!(held.is_dir());
    std::fs::remove_file(&directory).unwrap();
    std::fs::write(&directory, b"not a directory").unwrap();
    assert_ne!(remove(&mut error), 0);
    assert_eq!(std::fs::read(&directory).unwrap(), b"not a directory");
    std::fs::remove_dir_all(&scratch).unwrap();
}

#[test]
fn native_identifier_is_a_bounded_ascii_component() {
    unsafe extern "C" {
        fn dyn_ane_dense_identifier_component_valid(bytes: *const u8, length: usize) -> c_int;
    }
    let valid = |bytes: &[u8]| unsafe {
        dyn_ane_dense_identifier_component_valid(bytes.as_ptr(), bytes.len()) != 0
    };
    assert!(valid(b"0123456789abcdefABCDEF"));
    assert!(valid(b"01234567-89AB_CDEF.ane"));
    assert!(valid(&[b'a'; 255]));
    for rejected in [
        b"".as_slice(),
        b".",
        b"..",
        b"../abc",
        b"/abc",
        b"abc/def",
        b"abc\\def",
        b"abc:def",
        b"abc\0def",
        b"abc\ndef",
        "abc\u{e9}".as_bytes(),
        &[b'a'; 256],
    ] {
        assert!(!valid(rejected), "accepted {rejected:?}");
    }
    assert_eq!(
        unsafe { dyn_ane_dense_identifier_component_valid(std::ptr::null(), 1) },
        0
    );
}
