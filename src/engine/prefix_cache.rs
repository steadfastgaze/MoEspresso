//! Bounded exact-prefix conversation state. Expert weights remain Engine-owned.

use std::collections::VecDeque;
use std::rc::Rc;

use anyhow::{Context, Result, ensure};

use super::{Engine, Session, SessionSnapshot};

const MAX_ENTRIES: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum FrontierKind {
    Anchor,
    Prompt,
    Generated,
}

struct Entry {
    snapshot: SessionSnapshot,
    retained_bytes: u64,
    kind: FrontierKind,
}

/// Exact committed states, bounded by host allocation capacities and entry count.
/// A conversation retains a prompt anchor and its latest generated frontier
/// when they fit. Independent token histories compete only through LRU eviction.
pub struct PromptPrefixCache {
    owner: Rc<()>,
    budget_bytes: u64,
    entries: VecDeque<Entry>,
}

/// Restore before consuming a suffix. Complete hits use the saved last logits.
pub struct PrefixHit<'a> {
    snapshot: &'a SessionSnapshot,
    complete: bool,
}

impl PrefixHit<'_> {
    pub fn snapshot(&self) -> &SessionSnapshot {
        self.snapshot
    }

    pub fn matched_tokens(&self) -> usize {
        self.snapshot.frontier()
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    pub fn last_logits_bf16(&self) -> Option<&[u16]> {
        self.complete.then(|| self.snapshot.last_logits_bf16())
    }
}

fn exact_prefix_match(saved: &[u32], requested: &[u32]) -> bool {
    !saved.is_empty() && requested.starts_with(saved)
}

impl PromptPrefixCache {
    pub fn new(engine: &Engine, budget_bytes: u64) -> Self {
        Self {
            owner: engine.cache_identity.clone(),
            budget_bytes,
            entries: VecDeque::new(),
        }
    }

    pub fn retained_bytes(&self) -> Result<u64> {
        self.entries.iter().try_fold(0u64, |sum, entry| {
            sum.checked_add(entry.retained_bytes)
                .context("prompt cache retained byte count overflow")
        })
    }

    /// Release retained snapshots without changing cache ownership.
    #[cfg(test)]
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    /// Select the longest exact prefix and refresh its recency. Different
    /// histories never match through a shared session label or cache slot.
    pub fn best_match<'a>(
        &'a mut self,
        engine: &Engine,
        rendered_tokens: &[u32],
    ) -> Option<PrefixHit<'a>> {
        if !Rc::ptr_eq(&self.owner, &engine.cache_identity)
            || rendered_tokens.len() > engine.max_context
        {
            return None;
        }
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                exact_prefix_match(entry.snapshot.consumed_tokens(), rendered_tokens)
            })
            .max_by_key(|(_, entry)| entry.snapshot.frontier())
            .map(|(index, _)| index)?;
        let entry = self.entries.remove(index)?;
        self.entries.push_back(entry);
        let snapshot = &self.entries.back()?.snapshot;
        Some(PrefixHit {
            snapshot,
            complete: snapshot.frontier() == rendered_tokens.len(),
        })
    }

    pub fn offer_prompt_checkpoint(
        &mut self,
        engine: &Engine,
        session: &Session,
        full_rendered_tokens: &[u32],
    ) -> Result<bool> {
        ensure!(
            full_rendered_tokens.len() <= engine.max_context
                && exact_prefix_match(&session.consumed_tokens, full_rendered_tokens),
            "prompt checkpoint tokens differ from rendered request"
        );
        let kind = if session.frontier == full_rendered_tokens.len() {
            FrontierKind::Prompt
        } else {
            FrontierKind::Anchor
        };
        self.offer(engine, session, kind)
    }

    /// Only tokens already consumed by the engine are published. The final
    /// sampled token may not have been consumed and must be replayed next turn.
    pub(crate) fn offer_generated_checkpoint(
        &mut self,
        engine: &Engine,
        session: &Session,
        prompt: &[u32],
    ) -> Result<bool> {
        ensure!(
            exact_prefix_match(prompt, &session.consumed_tokens),
            "generated checkpoint differs from request lineage"
        );
        if session.frontier <= prompt.len() {
            return Ok(false);
        }
        self.offer(engine, session, FrontierKind::Generated)
    }

    fn offer(&mut self, engine: &Engine, session: &Session, kind: FrontierKind) -> Result<bool> {
        ensure!(
            Rc::ptr_eq(&self.owner, &engine.cache_identity),
            "prompt cache belongs to another engine"
        );
        engine.validate_snapshot_source(session)?;
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.snapshot.consumed_tokens() == session.consumed_tokens)
        {
            let entry = self.entries.remove(index).unwrap();
            self.entries.push_back(entry);
            return Ok(false);
        }
        let minimum = engine.snapshot_minimum_retained_bytes(session)?;
        if minimum > self.budget_bytes {
            return Ok(false);
        }
        // Replace earlier checkpoints of this chain, retaining distinct prompt
        // anchors for clients that rerender the assistant's generated text.
        self.entries.retain(|entry| {
            !(session
                .consumed_tokens
                .starts_with(entry.snapshot.consumed_tokens())
                && (entry.kind == kind
                    || (kind == FrontierKind::Prompt && entry.kind == FrontierKind::Generated)))
        });
        loop {
            let available = self.budget_bytes - self.retained_bytes()?;
            if self.entries.len() < MAX_ENTRIES
                && minimum <= available
                && let Some(snapshot) = engine.snapshot_with_budget(session, available)?
            {
                let retained_bytes = snapshot.retained_capacity_bytes()?;
                ensure!(
                    retained_bytes <= available,
                    "snapshot exceeded cache admission"
                );
                self.entries.push_back(Entry {
                    snapshot,
                    retained_bytes,
                    kind,
                });
                return Ok(true);
            }
            // Keep one completed prompt per conversation ahead of additional
            // anchors and generated frontiers. Extra state from a title request
            // must not evict another conversation's only reusable prompt.
            let extra = self
                .entries
                .iter()
                .position(|entry| entry.kind != FrontierKind::Prompt);
            if let Some(index) = extra {
                self.entries.remove(index);
            } else if kind == FrontierKind::Prompt && !self.entries.is_empty() {
                self.entries.pop_front();
            } else {
                return Ok(false);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_prefix_does_not_depend_on_chunk_alignment() {
        assert!(exact_prefix_match(&[7, 8, 9], &[7, 8, 9]));
        assert!(exact_prefix_match(&[7, 8, 9], &[7, 8, 9, 10]));
        assert!(!exact_prefix_match(&[7, 8, 9], &[7, 8]));
        assert!(!exact_prefix_match(&[7, 8, 9], &[7, 9, 8, 10]));
        assert!(!exact_prefix_match(&[], &[7]));
    }
}
