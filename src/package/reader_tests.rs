use super::*;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

impl ExpertLayer {
    pub(crate) fn at_eof_for_test(&self) -> Result<Self> {
        let mut layer = self.clone();
        layer.data_start = layer.file.metadata()?.len();
        Ok(layer)
    }
}

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "moespresso-reader-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn layer(&self, index: usize) -> ExpertLayer {
        let path = self.0.join(format!("layer-{index}"));
        if !path.exists() {
            let mut file = File::create(&path).unwrap();
            file.write_all(&(0..61).map(|i| i as u8).collect::<Vec<_>>())
                .unwrap();
        }
        let file = File::open(&path).unwrap();
        ExpertLayer {
            file_identity: file_identity(&file.metadata().unwrap()),
            file: Arc::new(file),
            path,
            layer: index,
            num_experts: 4,
            row_bytes: 12,
            data_start: 13,
            projections: std::array::from_fn(|i| ProjectionLayout {
                codec: "iq2_k".into(),
                out_features: 1,
                logical_in_features: 256,
                stored_in_features: 256,
                offset: i * 4,
                nbytes: 4,
            }),
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn positional_rows_match_independent_allocating_reader_and_leave_cursor() {
    let fixture = Fixture::new();
    let layer = fixture.layer(0);
    let mut cursor = &*layer.file;
    cursor.seek(SeekFrom::Start(7)).unwrap();
    let workspace = ExpertRowWorkspace::new(24).unwrap();
    for id in [3, 0, 2, 1, 3, 0] {
        let mut row = workspace.row(12).unwrap();
        layer.read_row_into(id, &mut row).unwrap();
        assert_eq!(&*row, layer.read_row(id).unwrap());
        assert_eq!(cursor.stream_position().unwrap(), 7);
    }
}

#[test]
fn invalid_ranges_and_short_reads_fail_without_a_complete_row() {
    let fixture = Fixture::new();
    let mut layer = fixture.layer(0);
    let mut row = [0xaa; 12];
    assert!(layer.read_row_into(4, &mut row).is_err());
    assert_eq!(row, [0xaa; 12]);
    assert!(layer.read_row_into(0, &mut row[..11]).is_err());
    assert_eq!(row, [0xaa; 12]);
    layer.data_start = u64::MAX - 1;
    assert!(layer.read_row_into(0, &mut row).is_err());
    assert_eq!(row, [0xaa; 12]);
    layer.data_start = 13;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&layer.path)
        .unwrap()
        .set_len(60)
        .unwrap();
    assert!(layer.read_row_into(3, &mut row).is_err());
    assert_eq!(&row[..11], &(49..60).collect::<Vec<u8>>());
    // A subsequent complete row overwrites every byte of the reusable staging.
    layer.read_row_into(0, &mut row).unwrap();
    assert_eq!(&row, &(13..25).collect::<Vec<u8>>()[..]);
}

#[test]
fn workspace_handles_share_one_bounded_allocation_and_recover_after_conflict() {
    let owner = ExpertRowWorkspace::new(24).unwrap();
    let handles = vec![owner.clone(); 48];
    let pointer = owner.row(24).unwrap().as_ptr();
    for handle in &handles {
        assert_eq!(handle.row(12).unwrap().as_ptr(), pointer);
    }
    let borrow = handles[0].row(24).unwrap();
    assert!(handles[47].row(12).is_err());
    drop(borrow);
    assert!(owner.row(25).is_err());
    assert_eq!(owner.row(24).unwrap().as_ptr(), pointer);
    assert_eq!(std::rc::Rc::strong_count(&owner.0), 49);
    let weak = std::rc::Rc::downgrade(&owner.0);
    drop(handles);
    drop(owner);
    assert!(weak.upgrade().is_none());
}

#[test]
fn forty_eight_descriptors_are_shared_with_prefill_and_dropped_with_owners() {
    let fixture = Fixture::new();
    let layers: Vec<_> = (0..48).map(|i| fixture.layer(i)).collect();
    let weak: Vec<_> = layers
        .iter()
        .map(|layer| Arc::downgrade(&layer.file))
        .collect();
    let prefill = layers.clone();
    for (decode, batch) in layers.iter().zip(&prefill) {
        assert!(Arc::ptr_eq(&decode.file, &batch.file));
        assert_eq!(Arc::strong_count(&decode.file), 2);
    }
    drop(layers);
    assert!(weak.iter().all(|file| file.upgrade().is_some()));
    drop(prefill);
    assert!(weak.iter().all(|file| file.upgrade().is_none()));
}

#[test]
fn atomic_path_replacement_cannot_pair_different_validated_files() {
    let fixture = Fixture::new();
    let layer = fixture.layer(0);
    assert!(layer.same_source(&fixture.layer(0)));
    std::fs::rename(&layer.path, fixture.0.join("retained")).unwrap();
    let replacement = fixture.layer(0);
    assert!(!layer.same_source(&replacement));
    assert!(layer.same_source(&layer.clone()));
    assert!(layer.read_row(0).is_err());
    let mut row = [0; 12];
    layer.read_row_into(0, &mut row).unwrap();
    assert_eq!(
        row,
        <[u8; 12]>::try_from((13..25).collect::<Vec<u8>>()).unwrap()
    );
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE; CPU storage check"]
fn package_iq3_and_iq2_rows_match_allocating_reader() -> Result<()> {
    let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
    let layers = [package.expert_layer(0)?, package.expert_layer(2)?];
    let workspace =
        ExpertRowWorkspace::new(layers.iter().map(|layer| layer.row_bytes).max().unwrap())?;
    for layer in layers {
        for id in [511, 0, 257, 1, 510, 2, 511, 0] {
            let expected = layer.read_row(id)?;
            let mut actual = workspace.row(layer.row_bytes)?;
            layer.read_row_into(id, &mut actual)?;
            assert_eq!(&*actual, expected);
        }
    }
    Ok(())
}
