use super::*;
use crate::package::ProjectionLayout;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

struct Fixture {
    root: PathBuf,
    layer: ExpertLayer,
}

impl Fixture {
    fn new() -> Self {
        Self::with_experts(4)
    }

    fn with_experts(experts: usize) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "moespresso-prefill-read-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("rows");
        std::fs::write(
            &path,
            (0..experts * 16).map(|n| n as u8).collect::<Vec<_>>(),
        )
        .unwrap();
        let file = File::open(&path).unwrap();
        let layer = ExpertLayer {
            file_identity: crate::package::file_identity(&file.metadata().unwrap()),
            file: Arc::new(file),
            path,
            layer: 0,
            num_experts: experts,
            row_bytes: 16,
            data_start: 0,
            projections: std::array::from_fn(|_| ProjectionLayout {
                codec: "iq2_k".into(),
                out_features: 1,
                logical_in_features: 256,
                stored_in_features: 256,
                offset: 0,
                nbytes: 4,
            }),
        };
        Self { root, layer }
    }

    fn truncate(&self, bytes: u64) {
        std::fs::OpenOptions::new()
            .write(true)
            .open(&self.layer.path)
            .unwrap()
            .set_len(bytes)
            .unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

fn pointers(workspace: &ExpertRowWorkspace) -> Vec<usize> {
    let storage = workspace.0.borrow();
    let mut pointers = vec![storage.current.as_ptr() as usize];
    pointers.extend(
        storage
            .reader
            .as_ref()
            .unwrap()
            .spare
            .iter()
            .map(|row| row.as_ptr() as usize),
    );
    pointers.sort_unstable();
    pointers
}

fn check_rows(workspace: &ExpertRowWorkspace, layer: &ExpertLayer, ids: &[u32]) {
    let mut rows = workspace
        .prefill_rows(layer, ids.to_vec())
        .unwrap()
        .unwrap();
    for &id in ids {
        assert_eq!(rows.next(id).unwrap(), layer.read_row(id as usize).unwrap());
    }
    rows.finish().unwrap();
}

#[test]
fn spawn_failure_keeps_one_owner_row_and_repeated_serial_reads() {
    let fixture = Fixture::new();
    let mut attempted = false;
    let workspace = ExpertRowWorkspace::with_reader_factory(32, |bytes| {
        ReadAhead::with_spawn(bytes, 4, 8, |jobs, replies| {
            attempted = true;
            assert!(matches!(jobs.try_recv(), Err(mpsc::TryRecvError::Empty)));
            drop(jobs);
            drop(replies);
            Err(std::io::Error::other(
                "injected prefill reader spawn failure",
            ))
        })
    })
    .unwrap();
    assert!(attempted);
    assert!(workspace.0.borrow().reader.is_none());
    assert_eq!(workspace.0.borrow().current.len(), 32);
    let current = workspace.0.borrow().current.as_ptr();
    let handles = vec![workspace.clone(); 48];
    for handle in &handles {
        assert!(
            handle
                .prefill_rows(&fixture.layer, vec![0, 1])
                .unwrap()
                .is_none()
        );
        for expert in [3, 0, 2] {
            let mut row = handle.row(fixture.layer.row_bytes).unwrap();
            fixture.layer.read_row_into(expert, &mut row).unwrap();
            assert_eq!(&*row, fixture.layer.read_row(expert).unwrap());
            assert_eq!(row.as_ptr(), current);
        }
    }
    assert!(workspace.0.borrow().reader.is_none());
}

#[test]
fn partial_worker_startup_failure_joins_started_readers() {
    let (stopped, completed) = mpsc::channel();
    let mut attempts = 0;
    let workspace = ExpertRowWorkspace::with_reader_factory(32, |bytes| {
        ReadAhead::with_spawn(bytes, 4, 8, |jobs, replies| {
            attempts += 1;
            if attempts == 3 {
                return Err(std::io::Error::other("injected partial startup failure"));
            }
            let stopped = stopped.clone();
            thread::Builder::new().spawn(move || {
                worker_loop(jobs, replies);
                stopped.send(()).unwrap();
            })
        })
    })
    .unwrap();
    assert_eq!(attempts, 3);
    assert!(workspace.0.borrow().reader.is_none());
    for _ in 0..2 {
        completed.recv_timeout(Duration::from_secs(2)).unwrap();
    }
    assert!(workspace.row(32).is_ok());
}

#[test]
fn fixed_global_allocations_move_in_source_order_and_return_before_reuse() {
    let fixture = Fixture::new();
    let workspace = ExpertRowWorkspace::with_prefill_reader_for_test(32, true).unwrap();
    let handles = vec![workspace.clone(); 48];
    let initial = pointers(&workspace);
    assert_eq!(initial.len(), PREFILL_READ_ROWS + 1);
    assert!(initial.windows(2).all(|pair| pair[0] != pair[1]));
    let mut cursor = handles[17]
        .prefill_rows(&fixture.layer, vec![0, 2, 3])
        .unwrap()
        .unwrap();
    for id in [0, 2, 3] {
        let row = cursor.next(id).unwrap();
        assert_eq!(row, fixture.layer.read_row(id as usize).unwrap());
        assert!(initial.contains(&(row.as_ptr() as usize)));
    }
    cursor.finish().unwrap();
    assert_eq!(pointers(&workspace), initial);
    for handle in &handles {
        assert!(Rc::ptr_eq(&workspace.0, &handle.0));
    }
    let storage = workspace.0.borrow();
    let reader = storage.reader.as_ref().unwrap();
    assert_eq!((reader.submitted, reader.returned), (2, 2));
    assert!(reader.pending.is_empty());
    assert_eq!(storage.current.len(), 32);
    assert!(reader.spare.iter().all(|row| row.len() == 32));
}

#[test]
fn complete_batches_recycle_each_bounded_window_without_reordering() {
    let fixture = Fixture::with_experts(64);
    let ids = (0..64).collect::<Vec<_>>();
    for (workers, window) in [(1, 1), (4, 4), (4, 8), (4, 24)] {
        let workspace = ExpertRowWorkspace::with_reader_factory(32, |bytes| {
            ReadAhead::with_layout(bytes, workers, window)
        })
        .unwrap();
        let initial = pointers(&workspace);
        for _ in 0..3 {
            check_rows(&workspace, &fixture.layer, &ids);
            assert_eq!(pointers(&workspace), initial);
            workspace.check_idle().unwrap();
        }
        let storage = workspace.0.borrow();
        let reader = storage.reader.as_ref().unwrap();
        assert_eq!(reader.spare.len(), window);
        assert_eq!((reader.submitted, reader.returned), (189, 189));
    }
}

#[test]
fn later_reader_runs_while_earlier_source_waits_and_consumption_stays_ordered() {
    let fixture = Fixture::new();
    let (started, entered) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let mut reader = ReadAhead::with_layout(32, 4, 4).unwrap();
    reader.next_fault = Some(TestFault::Pause {
        started,
        release: released,
    });
    reader.start(ReadSource::from(&fixture.layer), 1).unwrap();
    reader.start(ReadSource::from(&fixture.layer), 2).unwrap();
    entered.recv_timeout(Duration::from_secs(2)).unwrap();
    // Worker 1 completes independently of worker 0's blocked earlier row.
    // Put its reply back through a test-only channel so ordinary consumption
    // still exercises the same ordered pending queue.
    let reply = reader.workers[1]
        .replies
        .recv_timeout(Duration::from_secs(2))
        .unwrap();
    assert_eq!(reply.expert, 2);
    let (replay, replies) = mpsc::channel();
    replay.send(reply).unwrap();
    reader.workers[1].replies = replies;
    release.send(()).unwrap();
    for id in [1, 2] {
        let row = reader.complete(id).unwrap();
        assert_eq!(&row[..16], fixture.layer.read_row(id as usize).unwrap());
        reader.spare.push(row);
    }
    assert!(reader.pending.is_empty());
}

#[test]
fn zero_one_miss_and_serial_control_never_submit() {
    let fixture = Fixture::new();
    for pipeline in [false, true] {
        let workspace = ExpertRowWorkspace::with_prefill_reader_for_test(32, pipeline).unwrap();
        let initial = pointers(&workspace);
        for ids in [vec![], vec![3]] {
            assert!(
                workspace
                    .prefill_rows(&fixture.layer, ids)
                    .unwrap()
                    .is_none()
            );
        }
        if !pipeline {
            assert!(
                workspace
                    .prefill_rows(&fixture.layer, vec![0, 1, 2])
                    .unwrap()
                    .is_none()
            );
        }
        let mut row = workspace.row(16).unwrap();
        fixture.layer.read_row_into(3, &mut row).unwrap();
        drop(row);
        assert_eq!(pointers(&workspace), initial);
        let storage = workspace.0.borrow();
        let reader = storage.reader.as_ref().unwrap();
        assert_eq!((reader.submitted, reader.returned), (0, 0));
        assert_eq!(reader.workers.len(), PREFILL_READ_WORKERS);
    }
    let serial = ExpertRowWorkspace::new(32).unwrap();
    assert!(serial.0.borrow().reader.is_none());
    assert!(
        serial
            .prefill_rows(&fixture.layer, vec![0, 1])
            .unwrap()
            .is_none()
    );
}

#[test]
fn future_eof_is_delivered_at_its_row_and_healthy_worker_retries() {
    let fixture = Fixture::new();
    fixture.truncate(40);
    let workspace = ExpertRowWorkspace::with_prefill_reader_for_test(32, true).unwrap();
    let initial = pointers(&workspace);
    let mut rows = workspace
        .prefill_rows(&fixture.layer, vec![0, 1, 2])
        .unwrap()
        .unwrap();
    let mut published = Vec::new();
    for id in [0, 1] {
        assert_eq!(
            rows.next(id).unwrap(),
            fixture.layer.read_row(id as usize).unwrap()
        );
        published.push(id);
    }
    assert!(rows.next(2).is_err());
    assert_eq!(published, [0, 1]);
    drop(rows);
    assert!(workspace.0.borrow().reader.as_ref().unwrap().available());
    let after = pointers(&workspace);
    assert!(after.into_iter().all(|ptr| initial.contains(&ptr)));
    std::fs::write(&fixture.layer.path, (0..64).collect::<Vec<u8>>()).unwrap();
    check_rows(&workspace, &fixture.layer, &[0, 1, 2, 3]);
}

#[test]
fn first_row_error_submits_nothing_and_releases_owner_borrow() {
    let fixture = Fixture::new();
    fixture.truncate(4);
    let workspace = ExpertRowWorkspace::with_prefill_reader_for_test(32, true).unwrap();
    let mut rows = workspace
        .prefill_rows(&fixture.layer, vec![0, 1])
        .unwrap()
        .unwrap();
    assert!(rows.next(0).is_err());
    drop(rows);
    assert_eq!(workspace.0.borrow().reader.as_ref().unwrap().submitted, 0);
    assert!(workspace.row(16).is_ok());
}

#[test]
fn owner_error_wins_while_future_read_error_is_drained() {
    let fixture = Fixture::new();
    let workspace = ExpertRowWorkspace::with_prefill_reader_for_test(32, true).unwrap();
    workspace.fail_next_prefetch_for_test();
    let result = (|| -> Result<()> {
        let mut rows = workspace.prefill_rows(&fixture.layer, vec![0, 1])?.unwrap();
        rows.next(0)?;
        anyhow::bail!("owner error before future row");
    })();
    assert_eq!(
        result.unwrap_err().to_string(),
        "owner error before future row"
    );
    let storage = workspace.0.borrow();
    let reader = storage.reader.as_ref().unwrap();
    assert!(reader.available() && reader.pending.is_empty() && !reader.spare.is_empty());
    assert_eq!((reader.submitted, reader.returned), (1, 1));
    drop(storage);
    check_rows(&workspace, &fixture.layer, &[0, 1]);
}

#[test]
fn panic_or_result_disconnect_joins_and_future_groups_are_serial() {
    let fixture = Fixture::new();
    for fault in [TestFault::Panic, TestFault::Disconnect] {
        let workspace = ExpertRowWorkspace::with_prefill_reader_for_test(32, true).unwrap();
        let current = workspace.0.borrow().current.as_ptr() as usize;
        workspace.0.borrow_mut().reader.as_mut().unwrap().next_fault = Some(fault);
        let mut rows = workspace
            .prefill_rows(&fixture.layer, vec![0, 1, 2])
            .unwrap()
            .unwrap();
        assert_eq!(rows.next(0).unwrap(), fixture.layer.read_row(0).unwrap());
        assert!(rows.next(1).is_err());
        drop(rows);
        {
            let storage = workspace.0.borrow();
            let reader = storage.reader.as_ref().unwrap();
            assert!(!reader.available());
            assert!(
                reader.workers.is_empty() && reader.pending.is_empty() && reader.spare.is_empty()
            );
            assert_eq!(storage.current.as_ptr() as usize, current);
        }
        assert!(
            workspace
                .prefill_rows(&fixture.layer, vec![0, 1])
                .unwrap()
                .is_none()
        );
        let mut row = workspace.row(16).unwrap();
        fixture.layer.read_row_into(3, &mut row).unwrap();
        assert_eq!(&*row, fixture.layer.read_row(3).unwrap());
    }
}

#[test]
fn request_disconnect_is_deferred_to_the_future_source() {
    let fixture = Fixture::new();
    let workspace = ExpertRowWorkspace::with_prefill_reader_for_test(32, true).unwrap();
    {
        let mut storage = workspace.0.borrow_mut();
        let reader = storage.reader.as_mut().unwrap();
        let worker = &mut reader.workers[0];
        worker.sender.take();
        worker.worker.take().unwrap().join().unwrap();
        let (sender, receiver) = mpsc::channel();
        drop(receiver);
        worker.sender = Some(sender);
    }
    let mut rows = workspace
        .prefill_rows(&fixture.layer, vec![0, 1])
        .unwrap()
        .unwrap();
    assert_eq!(rows.next(0).unwrap(), fixture.layer.read_row(0).unwrap());
    assert!(rows.next(1).is_err());
    drop(rows);
    assert!(
        workspace
            .prefill_rows(&fixture.layer, vec![0, 1])
            .unwrap()
            .is_none()
    );
    assert!(workspace.row(16).is_ok());
}

#[test]
fn later_submit_failure_preserves_earlier_successful_publication() {
    let fixture = Fixture::new();
    let workspace = ExpertRowWorkspace::with_prefill_reader(32).unwrap();
    {
        let mut storage = workspace.0.borrow_mut();
        let worker = &mut storage.reader.as_mut().unwrap().workers[1];
        worker.sender.take();
        worker.worker.take().unwrap().join().unwrap();
        let (sender, receiver) = mpsc::channel();
        drop(receiver);
        worker.sender = Some(sender);
    }
    let mut rows = workspace
        .prefill_rows(&fixture.layer, vec![0, 1, 2, 3])
        .unwrap()
        .unwrap();
    for id in [0, 1] {
        assert_eq!(
            rows.next(id).unwrap(),
            fixture.layer.read_row(id as usize).unwrap()
        );
    }
    assert!(rows.next(2).is_err());
    drop(rows);
    assert!(
        workspace
            .prefill_rows(&fixture.layer, vec![0, 1])
            .unwrap()
            .is_none()
    );
    assert!(workspace.row(16).is_ok());
}

#[test]
fn group_drop_and_owner_error_wait_for_future_then_allow_retry() {
    for owner_error in [false, true] {
        let (started, entered) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let (dropping, drop_started) = mpsc::channel();
        let (done, completed) = mpsc::channel();
        let owner = thread::spawn(move || {
            let fixture = Fixture::new();
            let workspace = ExpertRowWorkspace::with_prefill_reader_for_test(32, true).unwrap();
            workspace.0.borrow_mut().reader.as_mut().unwrap().next_fault = Some(TestFault::Pause {
                started,
                release: released,
            });
            let result = (|| -> Result<()> {
                let mut rows = workspace
                    .prefill_rows(&fixture.layer, vec![0, 1, 2, 3])?
                    .unwrap();
                rows.next(0)?;
                dropping.send(()).unwrap();
                if owner_error {
                    anyhow::bail!("injected owner copy failure");
                }
                drop(rows);
                Ok(())
            })();
            if owner_error {
                assert_eq!(
                    result.unwrap_err().to_string(),
                    "injected owner copy failure"
                );
            } else {
                result.unwrap();
            }
            {
                let storage = workspace.0.borrow();
                let reader = storage.reader.as_ref().unwrap();
                assert_eq!((reader.submitted, reader.returned), (3, 3));
                assert!(reader.pending.is_empty() && !reader.spare.is_empty());
            }
            check_rows(&workspace, &fixture.layer, &[1, 3]);
            drop(workspace);
            done.send(()).unwrap();
        });
        entered.recv_timeout(Duration::from_secs(2)).unwrap();
        drop_started.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(
            completed.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        release.send(()).unwrap();
        completed.recv_timeout(Duration::from_secs(2)).unwrap();
        owner.join().unwrap();
    }
}

#[test]
fn invalid_cursor_order_and_owner_conflict_fail_closed() {
    let fixture = Fixture::new();
    let workspace = ExpertRowWorkspace::with_prefill_reader_for_test(32, true).unwrap();
    assert!(workspace.prefill_rows(&fixture.layer, vec![2, 0]).is_err());
    assert!(
        workspace
            .prefill_rows(&fixture.layer, vec![0; 513])
            .is_err()
    );
    let mut rows = workspace
        .prefill_rows(&fixture.layer, vec![0, 2])
        .unwrap()
        .unwrap();
    assert!(workspace.row(16).is_err());
    assert!(rows.next(2).is_err());
    drop(rows);
    check_rows(&workspace, &fixture.layer, &[0, 2]);
}
