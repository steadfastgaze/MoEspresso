//! Ordered expert-row consumption with a bounded concurrent read-ahead window.

use super::{
    ExpertLayer, PREFILL_READ_ROWS, PREFILL_READ_STACK_BYTES, PREFILL_READ_WORKERS, read_expert_row,
};
use anyhow::{Context, Result, ensure};
use std::cell::{RefCell, RefMut};
use std::collections::VecDeque;
use std::fs::File;
use std::rc::Rc;
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};

#[derive(Clone)]
struct ReadSource {
    file: Arc<File>,
    experts: usize,
    row_bytes: usize,
    data_start: u64,
}

impl From<&ExpertLayer> for ReadSource {
    fn from(layer: &ExpertLayer) -> Self {
        Self {
            file: layer.file.clone(),
            experts: layer.num_experts,
            row_bytes: layer.row_bytes,
            data_start: layer.data_start,
        }
    }
}

impl ReadSource {
    fn read(&self, expert: u32, row: &mut [u8]) -> Result<()> {
        read_expert_row(
            &self.file,
            self.experts,
            self.row_bytes,
            self.data_start,
            expert as usize,
            row,
        )
    }
}

struct ReadJob {
    source: ReadSource,
    expert: u32,
    row: Box<[u8]>,
    #[cfg(test)]
    fault: Option<TestFault>,
}

struct ReadReply {
    expert: u32,
    row: Box<[u8]>,
    result: Result<()>,
}

enum Pending {
    Worker { expert: u32, worker: usize },
    Failed(u32, anyhow::Error),
}

impl Pending {
    fn expert(&self) -> u32 {
        match self {
            Self::Worker { expert, .. } | Self::Failed(expert, _) => *expert,
        }
    }
}

struct ReaderWorker {
    sender: Option<mpsc::Sender<ReadJob>>,
    replies: mpsc::Receiver<ReadReply>,
    worker: Option<JoinHandle<()>>,
}

struct ReadAhead {
    workers: Vec<ReaderWorker>,
    spare: Vec<Box<[u8]>>,
    pending: VecDeque<Pending>,
    next_worker: usize,
    broken: bool,
    #[cfg(test)]
    submitted: u64,
    #[cfg(test)]
    returned: u64,
    #[cfg(test)]
    next_fault: Option<TestFault>,
}

fn worker_loop(jobs: mpsc::Receiver<ReadJob>, replies: mpsc::Sender<ReadReply>) {
    while let Ok(mut job) = jobs.recv() {
        #[cfg(test)]
        if let Some(fault) = job.fault.take() {
            match fault {
                TestFault::Panic => panic!("injected prefill reader panic"),
                TestFault::Disconnect => break,
                TestFault::Eof => job.source.data_start = job.source.file.metadata().unwrap().len(),
                TestFault::Pause { started, release } => {
                    let _ = started.send(());
                    let _ = release.recv();
                }
            }
        }
        let result = match job.row.get_mut(..job.source.row_bytes) {
            Some(row) => job.source.read(job.expert, row),
            None => Err(anyhow::anyhow!("prefill worker row exceeds staging")),
        };
        if replies
            .send(ReadReply {
                expert: job.expert,
                row: job.row,
                result,
            })
            .is_err()
        {
            break;
        }
    }
}

impl ReadAhead {
    fn new(bytes: usize) -> Result<Self> {
        Self::with_layout(bytes, PREFILL_READ_WORKERS, PREFILL_READ_ROWS)
    }

    fn with_layout(bytes: usize, workers: usize, rows: usize) -> Result<Self> {
        Self::with_spawn(bytes, workers, rows, |jobs, replies| {
            thread::Builder::new()
                .name("moespresso-prefill-read".into())
                .stack_size(PREFILL_READ_STACK_BYTES)
                .spawn(move || worker_loop(jobs, replies))
        })
    }

    fn with_spawn(
        bytes: usize,
        workers: usize,
        rows: usize,
        mut spawn: impl FnMut(
            mpsc::Receiver<ReadJob>,
            mpsc::Sender<ReadReply>,
        ) -> std::io::Result<JoinHandle<()>>,
    ) -> Result<Self> {
        ensure!(
            bytes > 0 && workers > 0 && rows >= workers,
            "invalid prefill reader window"
        );
        let mut reader = Self {
            workers: Vec::with_capacity(workers),
            spare: (0..rows)
                .map(|_| vec![0; bytes].into_boxed_slice())
                .collect(),
            pending: VecDeque::with_capacity(rows),
            next_worker: 0,
            broken: false,
            #[cfg(test)]
            submitted: 0,
            #[cfg(test)]
            returned: 0,
            #[cfg(test)]
            next_fault: None,
        };
        for _ in 0..workers {
            // The owner can submit only the allocated rows. Unbounded channels
            // cannot accumulate additional payloads, and let workers return all
            // queued rows before shutdown joins them.
            let (sender, jobs) = mpsc::channel();
            let (replies, receiver) = mpsc::channel();
            let worker = spawn(jobs, replies).context("start bounded prefill reader")?;
            reader.workers.push(ReaderWorker {
                sender: Some(sender),
                replies: receiver,
                worker: Some(worker),
            });
        }
        Ok(reader)
    }

    fn available(&self) -> bool {
        !self.broken && !self.workers.is_empty()
    }

    // No shared storage may be reused until every worker is quiescent.
    // On transport loss the owner keeps its current row and disables prefetch.
    fn disable(&mut self) {
        self.broken = true;
        for worker in &mut self.workers {
            worker.sender.take();
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.worker.take() {
                let _ = handle.join();
            }
        }
        self.workers.clear();
        self.pending.clear();
        self.spare.clear();
    }

    fn start(&mut self, source: ReadSource, expert: u32) -> Result<()> {
        ensure!(self.available(), "prefill reader is disabled");
        let row = self
            .spare
            .pop()
            .context("prefill reader spare row missing")?;
        let job = ReadJob {
            source,
            expert,
            row,
            #[cfg(test)]
            fault: self.next_fault.take(),
        };
        let worker = self.next_worker;
        self.next_worker = (worker + 1) % self.workers.len();
        let sender = self.workers[worker]
            .sender
            .as_ref()
            .context("prefill reader is disabled")?;
        match sender.send(job) {
            Ok(()) => {
                #[cfg(test)]
                {
                    self.submitted = self.submitted.saturating_add(1);
                }
                self.pending.push_back(Pending::Worker { expert, worker });
            }
            Err(error) => {
                self.spare.push(error.0.row);
                self.broken = true;
                // Keep preceding replies available. A failed future source
                // must not move its error before earlier rows are consumed.
                self.pending.push_back(Pending::Failed(
                    expert,
                    anyhow::anyhow!("prefill reader request channel disconnected"),
                ));
            }
        }
        Ok(())
    }

    fn complete(&mut self, expert: u32) -> Result<Box<[u8]>> {
        let pending = self
            .pending
            .pop_front()
            .context("prefill reader has no future row")?;
        match pending {
            Pending::Failed(expected, error) => {
                self.disable();
                ensure!(expected == expert, "prefill failed-row order differs");
                Err(error)
            }
            Pending::Worker {
                expert: expected,
                worker,
            } => {
                if expected != expert {
                    self.disable();
                    anyhow::bail!("prefill future-row order differs");
                }
                let reply = {
                    match self.workers[worker].replies.recv() {
                        Ok(reply) => reply,
                        Err(_) => {
                            self.disable();
                            anyhow::bail!("prefill reader result channel disconnected");
                        }
                    }
                };
                #[cfg(test)]
                {
                    self.returned = self.returned.saturating_add(1);
                }
                if reply.expert != expert {
                    self.disable();
                    anyhow::bail!("prefill returned-row identity differs");
                }
                match reply.result {
                    Ok(()) => Ok(reply.row),
                    Err(error) => {
                        self.spare.push(reply.row);
                        Err(error)
                    }
                }
            }
        }
    }

    fn drain(&mut self) {
        while let Some(pending) = self.pending.front() {
            if let Ok(row) = self.complete(pending.expert()) {
                self.spare.push(row);
            }
        }
    }
}

impl Drop for ReadAhead {
    fn drop(&mut self) {
        self.drain();
        self.disable();
    }
}

pub(super) struct RowStorage {
    current: Box<[u8]>,
    reader: Option<ReadAhead>,
    #[cfg(test)]
    pipeline: bool,
}

/// One Engine-owned workspace; clones do not allocate rows or start workers.
#[derive(Clone)]
pub(crate) struct ExpertRowWorkspace(pub(super) Rc<RefCell<RowStorage>>);

impl ExpertRowWorkspace {
    /// Require no owner row borrow or outstanding read-ahead request.
    pub(crate) fn check_idle(&self) -> Result<()> {
        let storage = self
            .0
            .try_borrow_mut()
            .context("expert row workspace is already borrowed")?;
        ensure!(
            storage
                .reader
                .as_ref()
                .is_none_or(|reader| reader.pending.is_empty()),
            "expert row workspace has an outstanding read-ahead request"
        );
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn fail_next_prefetch_for_test(&self) {
        self.0.borrow_mut().reader.as_mut().unwrap().next_fault = Some(TestFault::Eof);
    }
    pub(crate) fn new(bytes: usize) -> Result<Self> {
        ensure!(bytes > 0, "expert row workspace is empty");
        Ok(Self(Rc::new(RefCell::new(RowStorage {
            current: vec![0; bytes].into_boxed_slice(),
            reader: None,
            #[cfg(test)]
            pipeline: false,
        }))))
    }

    pub(crate) fn with_prefill_reader(bytes: usize) -> Result<Self> {
        Self::with_reader_factory(bytes, ReadAhead::new)
    }

    fn with_reader_factory(
        bytes: usize,
        create_reader: impl FnOnce(usize) -> Result<ReadAhead>,
    ) -> Result<Self> {
        let workspace = Self::new(bytes)?;
        {
            let mut storage = workspace.0.borrow_mut();
            // Read-ahead is optional. Failed spawn drains workers and staging;
            // the retained current row remains usable with conservative admission.
            storage.reader = create_reader(bytes).ok();
            #[cfg(test)]
            {
                storage.pipeline = true;
            }
        }
        Ok(workspace)
    }

    #[cfg(test)]
    fn with_prefill_reader_for_test(bytes: usize, pipeline: bool) -> Result<Self> {
        let workspace = Self::with_prefill_reader(bytes)?;
        workspace.0.borrow_mut().pipeline = pipeline;
        Ok(workspace)
    }

    pub(crate) fn row(&self, bytes: usize) -> Result<RefMut<'_, [u8]>> {
        let storage = self
            .0
            .try_borrow_mut()
            .context("expert row workspace is already borrowed")?;
        ensure!(
            bytes <= storage.current.len(),
            "expert row exceeds owner workspace"
        );
        Ok(RefMut::map(storage, |storage| {
            &mut storage.current[..bytes]
        }))
    }

    pub(crate) fn prefill_rows(
        &self,
        layer: &ExpertLayer,
        missing: Vec<u32>,
    ) -> Result<Option<PrefillRows<'_>>> {
        ensure!(
            missing.len() <= 512,
            "prefill miss list exceeds source bound"
        );
        if missing.len() < 2 {
            return Ok(None);
        }
        let storage = self
            .0
            .try_borrow_mut()
            .context("expert row workspace is already borrowed")?;
        #[cfg(test)]
        if !storage.pipeline {
            return Ok(None);
        }
        if !storage.reader.as_ref().is_some_and(ReadAhead::available) {
            return Ok(None);
        }
        ensure!(
            layer.row_bytes <= storage.current.len(),
            "expert row exceeds owner workspace"
        );
        ensure!(
            missing.windows(2).all(|pair| pair[0] < pair[1]),
            "prefill misses are not source ordered"
        );
        Ok(Some(PrefillRows {
            storage,
            source: ReadSource::from(layer),
            missing,
            next: 0,
        }))
    }
}

/// The group guard drains its bounded window before the pool scope can abort.
pub(crate) struct PrefillRows<'a> {
    storage: RefMut<'a, RowStorage>,
    source: ReadSource,
    missing: Vec<u32>,
    next: usize,
}

impl PrefillRows<'_> {
    pub(crate) fn next(&mut self, expert: u32) -> Result<&[u8]> {
        ensure!(
            self.missing.get(self.next) == Some(&expert),
            "prefill row consumption order differs"
        );
        if self.next == 0 {
            self.source
                .read(expert, &mut self.storage.current[..self.source.row_bytes])?;
        } else {
            let mut ready = self
                .storage
                .reader
                .as_mut()
                .context("prefill reader missing")?
                .complete(expert)?;
            std::mem::swap(&mut self.storage.current, &mut ready);
            self.storage
                .reader
                .as_mut()
                .context("prefill reader missing")?
                .spare
                .push(ready);
        }
        self.next += 1;
        let reader = self
            .storage
            .reader
            .as_mut()
            .context("prefill reader missing")?;
        while reader.available() && !reader.spare.is_empty() {
            let Some(&next) = self.missing.get(self.next + reader.pending.len()) else {
                break;
            };
            reader.start(self.source.clone(), next)?;
        }
        Ok(&self.storage.current[..self.source.row_bytes])
    }

    pub(crate) fn finish(self) -> Result<()> {
        ensure!(
            self.next == self.missing.len(),
            "prefill row cursor was not consumed"
        );
        Ok(())
    }
}

impl Drop for PrefillRows<'_> {
    fn drop(&mut self) {
        if let Some(reader) = &mut self.storage.reader {
            reader.drain();
        }
    }
}

#[cfg(test)]
enum TestFault {
    Panic,
    Disconnect,
    Eof,
    Pause {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    },
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod idle_tests {
    use super::*;

    #[test]
    fn idle_check_rejects_owner_borrows_and_pending_read_requests() -> Result<()> {
        let workspace = ExpertRowWorkspace::new(16)?;
        workspace.check_idle()?;
        let row = workspace.row(16)?;
        assert!(workspace.check_idle().is_err());
        drop(row);
        workspace.check_idle()?;
        let read = workspace.0.borrow();
        assert!(workspace.check_idle().is_err());
        drop(read);
        let workspace = ExpertRowWorkspace::with_prefill_reader(16)?;
        workspace.check_idle()?;
        if let Some(reader) = workspace.0.borrow_mut().reader.as_mut() {
            reader
                .pending
                .push_back(Pending::Failed(0, anyhow::anyhow!("pending test reply")));
        } else {
            anyhow::bail!("test read-ahead worker was not created");
        }
        assert!(workspace.check_idle().is_err());
        workspace.0.borrow_mut().reader.as_mut().unwrap().drain();
        workspace.check_idle()?;
        Ok(())
    }
}
