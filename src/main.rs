//! Native text generation and loopback Chat Completions serving.

#[cfg(target_os = "macos")]
mod check_ane;
#[cfg(target_os = "macos")]
mod disk_config;
#[cfg(target_os = "macos")]
mod help;
#[cfg(target_os = "macos")]
mod speed;

#[cfg(target_os = "macos")]
mod cli {
    use crate::{help, speed};
    use std::io::{self, Write};

    use anyhow::{Context, Result, anyhow, bail, ensure};
    use moespresso::engine::{DiskCacheMode, Engine};
    use moespresso::generation::{GenerationOptions, generate};
    use moespresso::host_memory::{HostMemoryObservation, observe_host_memory};
    use moespresso::memory::ExpertCapacityControlBudget;
    use moespresso::memory::{PROCESS_OVERHEAD_ALLOWANCE_BYTES, PackageFootprint};
    use moespresso::metal::Metal;
    use moespresso::model::QwenConfig;
    use moespresso::package::Package;
    use moespresso::routing::RoutingMode;
    use moespresso::sampling::{Sampler, SamplingOptions};
    use moespresso::server::{ServeCacheOptions, ServeOptions, serve_with_prefill_capacity};
    use moespresso::tokenizer::{ChatOptions, ReasoningEffort, TextTokenizer};
    use rand_core::{OsRng, TryRngCore};
    use serde_json::json;

    const GIB_BYTES: u64 = 1 << 30;
    const OS_RESERVE_BYTES: u64 = 5 * GIB_BYTES;
    // Automatic startup headroom for other applications. These byte
    // allowances are policy choices, not guarantees against later pressure.
    const AUTO_LIVE_RESERVE_BYTES: u64 = 3 * GIB_BYTES;
    const AUTO_METAL_RESERVE_BYTES: u64 = GIB_BYTES;

    #[derive(Debug, PartialEq)]
    struct ServeArgs {
        package: String,
        port: u16,
        expert_capacity: Option<usize>,
        max_memory_bytes: Option<u64>,
        prompt_cache_bytes: Option<u64>,
        max_context: Option<usize>,
        thinking: bool,
        cache_routing: RoutingMode,
        adaptive_memory: bool,
        no_disk_cache: bool,
        disk_cache_dir: Option<std::path::PathBuf>,
        disk_cache_bytes: Option<u64>,
        disk_cache_mode: Option<DiskCacheMode>,
    }

    struct ResolvedCapacity {
        expert_capacity: usize,
        prefill_capacity: usize,
        control_budget: Option<ExpertCapacityControlBudget>,
        adaptive_memory: bool,
        ane_reserved_bytes: u64,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum AdaptiveMode {
        Enabled,
        Disabled,
        FullResident,
        BelowFloor,
    }

    impl AdaptiveMode {
        fn enabled(self) -> bool {
            self == Self::Enabled
        }

        fn reason(self) -> &'static str {
            match self {
                Self::Enabled => "automatic-bounded-serving",
                Self::Disabled => "static-capacity-requested",
                Self::FullResident => "fully-resident",
                Self::BelowFloor => "below-64-slot-adaptive-floor",
            }
        }
    }

    fn adaptive_mode(slots: usize, requested: bool) -> AdaptiveMode {
        if slots == 512 {
            AdaptiveMode::FullResident
        } else if !requested {
            AdaptiveMode::Disabled
        } else if slots < 64 {
            AdaptiveMode::BelowFloor
        } else {
            AdaptiveMode::Enabled
        }
    }

    #[derive(Debug, PartialEq)]
    struct GenerateArgs {
        package: String,
        prompt: String,
        thinking: bool,
        max_tokens: usize,
        expert_capacity: Option<usize>,
        max_memory_bytes: Option<u64>,
        max_context: usize,
        cache_routing: RoutingMode,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        min_p: Option<f64>,
        presence_penalty: Option<f64>,
        presence_context_size: Option<usize>,
        seed: Option<u64>,
    }

    #[derive(Debug, PartialEq)]
    enum Command {
        Help(&'static str),
        Version,
        CheckAne,
        Generate(GenerateArgs),
        Serve(ServeArgs),
        Speed(speed::SpeedOptions),
    }

    fn positive(value: String, flag: &str) -> Result<usize> {
        let number: usize = value
            .parse()
            .with_context(|| format!("{flag} requires a positive integer"))?;
        ensure!(number > 0, "{flag} requires a positive integer");
        Ok(number)
    }

    fn nonnegative(value: String, flag: &str) -> Result<usize> {
        value
            .parse()
            .with_context(|| format!("{flag} requires a nonnegative integer"))
    }

    fn finite(value: String, flag: &str) -> Result<f64> {
        let number: f64 = value
            .parse()
            .with_context(|| format!("{flag} requires a finite number"))?;
        ensure!(number.is_finite(), "{flag} requires a finite number");
        Ok(number)
    }

    fn gib_bytes(value: String, flag: &str) -> Result<u64> {
        let gib = finite(value, flag)?;
        ensure!(gib > 0.0, "{flag} must be positive");
        let bytes = gib * GIB_BYTES as f64;
        ensure!(
            bytes >= 1.0 && bytes < u64::MAX as f64,
            "{flag} is outside the supported byte range"
        );
        Ok(bytes as u64)
    }

    fn prompt_cache_gib_bytes(value: String) -> Result<u64> {
        if finite(value.clone(), "--prompt-cache-gb")? == 0.0 {
            return Ok(0);
        }
        gib_bytes(value, "--prompt-cache-gb")
    }

    fn serve_context(requested: Option<usize>, package_max: usize) -> Result<usize> {
        ensure!(
            (1..=262144).contains(&package_max),
            "unsupported package context limit"
        );
        let resolved = requested.unwrap_or(package_max.min(131072));
        ensure!(
            resolved > 0 && resolved <= package_max,
            "serve context must be positive and no greater than package limit {package_max}"
        );
        Ok(resolved)
    }

    fn serve_cache_bytes(explicit_capacity: Option<usize>, requested: Option<u64>) -> Result<u64> {
        if explicit_capacity.is_some() {
            ensure!(
                requested.unwrap_or(0) == 0,
                "positive --prompt-cache-gb conflicts with fixed --expert-capacity"
            );
            return Ok(0);
        }
        Ok(requested.unwrap_or(GIB_BYTES))
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct HostBudget {
        process_budget_bytes: u64,
        limiting_source: &'static str,
    }

    fn clamp_host_budget(
        mut budget: HostBudget,
        user_limit_bytes: Option<u64>,
    ) -> Result<HostBudget> {
        if let Some(limit) = user_limit_bytes {
            ensure!(limit > 0, "explicit memory limit must be positive");
            if limit <= budget.process_budget_bytes {
                budget.process_budget_bytes = limit;
                budget.limiting_source = "explicit-user-limit";
            }
        }
        Ok(budget)
    }

    fn host_budget(
        user_limit_bytes: u64,
        observed: HostMemoryObservation,
        metal_recommended_bytes: u64,
    ) -> Result<HostBudget> {
        clamp_host_budget(
            automatic_host_budget(observed, metal_recommended_bytes)?,
            Some(user_limit_bytes),
        )
    }

    fn automatic_host_budget(
        observed: HostMemoryObservation,
        metal_recommended_bytes: u64,
    ) -> Result<HostBudget> {
        fn leave(bytes: u64, reserve: u64, name: &str) -> Result<u64> {
            bytes
                .checked_sub(reserve)
                .filter(|&value| value > 0)
                .with_context(|| {
                    format!("automatic admission: {name} cannot leave {reserve} bytes of headroom")
                })
        }
        ensure!(
            observed.free_inactive_bytes <= observed.physical_ram_bytes,
            "free plus inactive memory exceeds installed RAM"
        );
        let candidates = [
            (
                "physical-minus-5-GiB",
                leave(
                    observed.physical_ram_bytes,
                    OS_RESERVE_BYTES,
                    "installed RAM",
                )?,
            ),
            (
                "free-plus-inactive-minus-3-GiB",
                leave(
                    observed.free_inactive_bytes,
                    AUTO_LIVE_RESERVE_BYTES,
                    "live free plus inactive memory",
                )?,
            ),
            (
                "Metal-recommended-minus-1-GiB",
                leave(
                    metal_recommended_bytes,
                    AUTO_METAL_RESERVE_BYTES,
                    "Metal recommended working set",
                )?,
            ),
        ];
        let (mut limiting_source, mut process_budget_bytes) = candidates[0];
        for (name, bytes) in candidates.into_iter().skip(1) {
            if bytes < process_budget_bytes {
                (limiting_source, process_budget_bytes) = (name, bytes);
            }
        }
        if let Some(wired) = observed.gpu_wired_limit_bytes {
            let bytes = leave(wired, AUTO_METAL_RESERVE_BYTES, "GPU wired limit")?;
            if bytes < process_budget_bytes {
                (limiting_source, process_budget_bytes) = ("GPU-wired-limit-minus-1-GiB", bytes);
            }
        }
        Ok(HostBudget {
            process_budget_bytes,
            limiting_source,
        })
    }

    fn operating_host_budget(
        observed: HostMemoryObservation,
        metal_recommended_bytes: u64,
        user_limit_bytes: Option<u64>,
    ) -> Result<HostBudget> {
        let metal_reserve = AUTO_METAL_RESERVE_BYTES;
        let leave = |bytes: u64, reserve: u64, name: &str| {
            bytes
                .checked_sub(reserve)
                .filter(|&value| value > 0)
                .with_context(|| {
                    format!("operating admission: {name} cannot leave {reserve} bytes of headroom")
                })
        };
        let candidates = [
            (
                "physical-minus-5-GiB",
                leave(
                    observed.physical_ram_bytes,
                    OS_RESERVE_BYTES,
                    "installed RAM",
                )?,
            ),
            (
                "Metal-recommended-minus-1-GiB",
                leave(
                    metal_recommended_bytes,
                    metal_reserve,
                    "Metal recommended working set",
                )?,
            ),
        ];
        let (mut limiting_source, mut process_budget_bytes) = candidates[0];
        for (source, bytes) in candidates.into_iter().skip(1) {
            if bytes < process_budget_bytes {
                (limiting_source, process_budget_bytes) = (source, bytes);
            }
        }
        if let Some(wired) = observed.gpu_wired_limit_bytes {
            let bytes = leave(wired, metal_reserve, "GPU wired limit")?;
            if bytes < process_budget_bytes {
                (limiting_source, process_budget_bytes) = ("GPU-wired-limit-minus-1-GiB", bytes);
            }
        }
        clamp_host_budget(
            HostBudget {
                process_budget_bytes,
                limiting_source,
            },
            user_limit_bytes,
        )
    }

    fn resolve_capacity(
        package: &Package,
        max_context: usize,
        explicit_capacity: Option<usize>,
        max_memory_bytes: Option<u64>,
        prompt_cache_bytes: u64,
        disk_snapshot_overlap_bytes: u64,
        adaptive_memory: bool,
    ) -> Result<ResolvedCapacity> {
        if max_context >= 4096 {
            let mut refusals = Vec::new();
            for physical_rows in [8192, 4096] {
                if max_context < physical_rows
                    || (physical_rows == 8192 && explicit_capacity.is_some_and(|slots| slots < 512))
                {
                    continue;
                }
                match resolve_capacity_with_ane(
                    package,
                    max_context,
                    explicit_capacity,
                    max_memory_bytes,
                    prompt_cache_bytes,
                    disk_snapshot_overlap_bytes,
                    adaptive_memory,
                    Some(physical_rows),
                ) {
                    Ok(capacity) => return Ok(capacity),
                    Err(error) => refusals.push(format!("{physical_rows} rows: {error:#}")),
                }
            }
            eprintln!(
                "⚠ ANE prefill cannot be admitted; using GPU prefill: {}",
                refusals.join("; ")
            );
        } else {
            eprintln!("⚠ ANE prefill requires at least 4096 context tokens; using GPU prefill");
        }
        resolve_capacity_with_ane(
            package,
            max_context,
            explicit_capacity,
            max_memory_bytes,
            prompt_cache_bytes,
            disk_snapshot_overlap_bytes,
            adaptive_memory,
            None,
        )
    }

    fn validate_ane_candidate(
        physical_rows: Option<usize>,
        expert_capacity: usize,
        workspace_rows: usize,
        context: usize,
    ) -> Result<()> {
        let Some(physical_rows) = physical_rows else {
            return Ok(());
        };
        ensure!(
            matches!(physical_rows, 4096 | 8192),
            "ANE physical rows must be 4096 or 8192"
        );
        ensure!(
            context >= physical_rows,
            "ANE shape exceeds the context limit"
        );
        ensure!(
            moespresso::prefill_capacity::ane_physical_rows(workspace_rows)? == physical_rows,
            "ANE reservation differs from the initial workspace shape"
        );
        if physical_rows == 8192 {
            ensure!(
                expert_capacity == 512 && workspace_rows == 8192,
                "8192-row ANE admission requires full residency and an 8192-row workspace"
            );
        } else {
            ensure!(
                matches!(workspace_rows, 2048 | 4096),
                "4096-row ANE admission requires an eligible ordinary or phased workspace"
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_capacity_with_ane(
        package: &Package,
        max_context: usize,
        explicit_capacity: Option<usize>,
        max_memory_bytes: Option<u64>,
        prompt_cache_bytes: u64,
        disk_snapshot_overlap_bytes: u64,
        adaptive_memory: bool,
        ane: Option<usize>,
    ) -> Result<ResolvedCapacity> {
        let ane_reserved_bytes = ane
            .map(moespresso::memory::ane_prefill_allowance_bytes)
            .transpose()?
            .unwrap_or(0);
        let process_allowance = PROCESS_OVERHEAD_ALLOWANCE_BYTES
            .checked_add(ane_reserved_bytes)
            .context("ANE admission allowance overflow")?;
        if let Some(capacity) = explicit_capacity {
            ensure!(
                max_memory_bytes.is_none(),
                "expert capacity and memory budget conflict"
            );
            ensure!(
                prompt_cache_bytes == 0 && disk_snapshot_overlap_bytes == 0,
                "enabled caches conflict with fixed --expert-capacity"
            );
            let select_rows = || -> Result<usize> {
                if max_context <= moespresso::prefill_capacity::DEFAULT_ROWS {
                    return Ok(moespresso::prefill_capacity::DEFAULT_ROWS);
                }
                let footprint = PackageFootprint::from_package(package)?;
                let recommended = Metal::new()?
                    .memory_stats()?
                    .device_recommended_max_working_set_bytes;
                let budget = automatic_host_budget(observe_host_memory()?, recommended)?;
                if let Some(rows) = ane {
                    validate_ane_candidate(ane, capacity, rows, max_context)?;
                    let wide = footprint.estimate_fixed_capacity_prefill(
                        max_context,
                        capacity,
                        rows,
                        process_allowance,
                    )?;
                    ensure!(
                        wide.accounted_component_sum_bytes <= budget.process_budget_bytes,
                        "fixed expert capacity leaves insufficient memory for ANE and {rows}-row prefill"
                    );
                    return Ok(rows);
                }
                let wide = footprint.estimate_fixed_capacity_prefill(
                    max_context,
                    capacity,
                    moespresso::prefill_capacity::AUTOMATIC_ROWS,
                    process_allowance,
                )?;
                Ok(moespresso::prefill_capacity::admitted_rows(
                    wide.accounted_component_sum_bytes,
                    budget.process_budget_bytes,
                ))
            };
            let rows = if ane.is_some() {
                select_rows()?
            } else {
                select_rows().unwrap_or_else(|error| {
                    eprintln!(
                        "prefill admission unavailable; retaining 512 rows for fixed expert capacity: {error:#}"
                    );
                    moespresso::prefill_capacity::DEFAULT_ROWS
                })
            };
            validate_ane_candidate(ane, capacity, rows, max_context)?;
            eprintln!(
                "prefill admission: mode=fixed-expert-capacity selected_slots={capacity} prefill_capacity={rows}"
            );
            return Ok(ResolvedCapacity {
                expert_capacity: capacity,
                prefill_capacity: rows,
                control_budget: None,
                adaptive_memory: false,
                ane_reserved_bytes,
            });
        }
        let footprint = PackageFootprint::from_package(package)?;
        let metal_recommended = Metal::new()?
            .memory_stats()?
            .device_recommended_max_working_set_bytes;
        let observed = observe_host_memory()?;
        let budget = match max_memory_bytes {
            Some(limit) => host_budget(limit, observed, metal_recommended)?,
            None => automatic_host_budget(observed, metal_recommended)?,
        };
        let user_limit_bytes = max_memory_bytes.unwrap_or(budget.process_budget_bytes);
        let (estimate, plan) = footprint.plan_ordinary_capacity_with_cache_budgets(
            max_context,
            budget.process_budget_bytes,
            user_limit_bytes,
            process_allowance,
            prompt_cache_bytes,
            disk_snapshot_overlap_bytes,
        )?;
        if ane == Some(8192) {
            ensure!(
                plan.slots == 512,
                "8192-row ANE admission requires full expert residency"
            );
        }
        let requested_rows = match ane {
            Some(8192) => 8192,
            Some(4096) if plan.slots == 512 || !adaptive_memory || plan.slots < 64 => 4096,
            _ => moespresso::prefill_capacity::AUTOMATIC_ROWS,
        };
        let (estimate, plan, prefill_capacity) =
            footprint.upgrade_prefill_capacity_to(max_context, estimate, plan, requested_rows)?;
        validate_ane_candidate(ane, plan.slots, prefill_capacity, max_context)?;
        let adaptive_mode = adaptive_mode(plan.slots, adaptive_memory);
        let control_budget = adaptive_mode
            .enabled()
            .then(|| ExpertCapacityControlBudget::from_admission(&footprint, &estimate, &plan))
            .transpose()?;
        let control_budget = if adaptive_mode.enabled() {
            let operating = operating_host_budget(observed, metal_recommended, max_memory_bytes)?;
            let budget = control_budget
                .context("adaptive memory requires budgeted bounded storage below 512 slots")?
                .with_operating_budget(
                    &footprint,
                    &estimate,
                    max_context,
                    prefill_capacity,
                    operating.process_budget_bytes,
                )?;
            eprintln!(
                "adaptive operating admission: initial_slots={} maximum_slots={} startup_budget={}B operating_budget={}B serving_peak={}B limiting_source={}",
                budget.initial_capacity(),
                budget.maximum_capacity(),
                budget.startup_budget_bytes(),
                budget.operating_budget_bytes(),
                budget.initial_serving_peak_bytes(),
                operating.limiting_source,
            );
            Some(budget)
        } else {
            control_budget
        };
        if ane.is_some() && prefill_capacity < 4096 {
            ensure!(
                prefill_capacity == moespresso::prefill_capacity::AUTOMATIC_ROWS,
                "ANE admission requires an eligible prefill workspace"
            );
            control_budget
                .context("ANE admission requires a wider adaptive prefill phase")?
                .validate_prefill_phase(&footprint, max_context, prefill_capacity, 4096)?;
        }
        let selected_expert_bytes = footprint
            .packed_expert_bytes_per_uniform_slot
            .checked_mul(u64::try_from(plan.slots)?)
            .context("selected expert payload byte count overflow")?;
        let expert_ceiling_surplus = estimate
            .expert_stream_payload_at_ceiling_bytes
            .checked_sub(selected_expert_bytes)
            .context("selected expert payload exceeds estimate ceiling")?;
        let selected_phase_peak = |peak: u64| {
            peak.checked_sub(expert_ceiling_surplus)
                .context("expert ceiling surplus exceeds phase peak")
        };
        let startup_peak = selected_phase_peak(estimate.startup_phase_peak_bytes)?;
        let construction_peak = selected_phase_peak(estimate.engine_construction_phase_peak_bytes)?;
        let serving_peak = selected_phase_peak(estimate.serving_phase_peak_bytes)?;
        eprintln!(
            "expert storage admission: metadata_reserve={}B initial_metadata_allocation={}B",
            estimate.expert_storage_metadata_reserve_bytes,
            footprint.expert_storage_metadata_bytes(plan.slots)?,
        );
        eprintln!(
            "memory admission: mode={} selected_slots={} accounted_components={}B effective_budget={}B limiting_source={} process_overhead_allowance={}B prompt_cache_retained={}B disk_snapshot_overlap={}B workspace_stage={:?} limiting_phase={:?} startup_peak={}B engine_construction_peak={}B serving_peak={}B",
            if max_memory_bytes.is_some() {
                "automatic-startup-capped"
            } else {
                "automatic-startup"
            },
            plan.slots,
            plan.accounted_peak_bytes,
            plan.effective_budget_bytes,
            budget.limiting_source,
            estimate.driver_allocator_allowance_bytes,
            estimate.prefix_retained_budget_bytes,
            estimate.snapshot_capture_restore_overlap_reserve_bytes,
            estimate.max_workspace_across_candidates.stage,
            estimate.limiting_phase,
            startup_peak,
            construction_peak,
            serving_peak,
        );
        eprintln!(
            "prefill admission: prefill_capacity={prefill_capacity} workspace_requested={}B remaining_budget={}B",
            estimate.prefill_workspace_requested_bytes, plan.remaining_budget_bytes
        );
        eprintln!(
            "adaptive memory: enabled={} reason={}",
            adaptive_mode.enabled(),
            adaptive_mode.reason()
        );
        if let Some(physical_rows) = ane {
            eprintln!(
                "ANE admission: retained_bytes={ane_reserved_bytes} physical_rows={physical_rows} minimum_actual_rows=4096"
            );
        }
        Ok(ResolvedCapacity {
            expert_capacity: plan.slots,
            prefill_capacity,
            control_budget,
            adaptive_memory: adaptive_mode.enabled(),
            ane_reserved_bytes,
        })
    }

    fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Command> {
        let mut args = args.into_iter();
        match args.next().as_deref() {
            Some("--help" | "-h") => return Ok(Command::Help(help::OVERVIEW)),
            Some("--version" | "-V") => return Ok(Command::Version),
            Some("check-ane") => {
                return match args.next().as_deref() {
                    None => Ok(Command::CheckAne),
                    Some("--help" | "-h") => Ok(Command::Help(help::CHECK_ANE)),
                    Some(arg) => bail!(
                        "unexpected check-ane argument: {arg}\nUse moespresso check-ane --help for options."
                    ),
                };
            }
            Some("serve") => return parse_serve_args(args),
            Some("speed") => {
                let options: Vec<_> = args.collect();
                if options
                    .iter()
                    .any(|option| option == "--help" || option == "-h")
                {
                    return Ok(Command::Help(help::SPEED));
                }
                return Ok(Command::Speed(speed::SpeedOptions::parse(
                    options.into_iter(),
                )?));
            }
            Some("generate") => {}
            _ => bail!("{}", help::OVERVIEW),
        }
        let mut package = None;
        let mut prompt = None;
        let mut thinking = true;
        let mut max_tokens = 128;
        let mut expert_capacity = None;
        let mut max_memory_bytes = None;
        let mut max_context = 4096;
        let mut cache_routing = RoutingMode::CachePrior;
        let mut temperature = None;
        let mut top_p = None;
        let mut top_k = None;
        let mut min_p = None;
        let mut presence_penalty = None;
        let mut presence_context_size = None;
        let mut seed = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => return Ok(Command::Help(help::GENERATE)),
                "--prompt" => {
                    ensure!(prompt.is_none(), "--prompt specified twice");
                    prompt = Some(args.next().context("--prompt requires text")?);
                }
                "--no-thinking" => {
                    ensure!(thinking, "--no-thinking specified twice");
                    thinking = false;
                }
                "--max-tokens" => {
                    max_tokens =
                        positive(args.next().context("--max-tokens requires a value")?, &arg)?;
                }
                "--expert-capacity" => {
                    ensure!(
                        expert_capacity.is_none() && max_memory_bytes.is_none(),
                        "--expert-capacity and --max-memory-gb are mutually exclusive"
                    );
                    expert_capacity = Some(positive(
                        args.next().context("--expert-capacity requires a value")?,
                        &arg,
                    )?);
                }
                "--max-memory-gb" => {
                    ensure!(
                        max_memory_bytes.is_none() && expert_capacity.is_none(),
                        "--max-memory-gb and --expert-capacity are mutually exclusive"
                    );
                    max_memory_bytes = Some(gib_bytes(
                        args.next().context("--max-memory-gb requires a value")?,
                        "--max-memory-gb",
                    )?);
                }
                "--max-context" => {
                    max_context =
                        positive(args.next().context("--max-context requires a value")?, &arg)?;
                }
                "--cache-routing" => {
                    cache_routing = match args.next().as_deref() {
                        Some("off") => RoutingMode::Unbiased,
                        Some("prefer-resident") => RoutingMode::CachePrior,
                        _ => bail!("--cache-routing must be off or prefer-resident"),
                    };
                }
                "--temperature" => {
                    ensure!(temperature.is_none(), "--temperature specified twice");
                    temperature = Some(finite(
                        args.next().context("--temperature requires a value")?,
                        &arg,
                    )?);
                }
                "--top-p" => {
                    ensure!(top_p.is_none(), "--top-p specified twice");
                    top_p = Some(finite(
                        args.next().context("--top-p requires a value")?,
                        &arg,
                    )?);
                }
                "--top-k" => {
                    ensure!(top_k.is_none(), "--top-k specified twice");
                    top_k = Some(nonnegative(
                        args.next().context("--top-k requires a value")?,
                        &arg,
                    )?);
                }
                "--min-p" => {
                    ensure!(min_p.is_none(), "--min-p specified twice");
                    min_p = Some(finite(
                        args.next().context("--min-p requires a value")?,
                        &arg,
                    )?);
                }
                "--presence-penalty" => {
                    ensure!(
                        presence_penalty.is_none(),
                        "--presence-penalty specified twice"
                    );
                    presence_penalty = Some(finite(
                        args.next().context("--presence-penalty requires a value")?,
                        &arg,
                    )?);
                }
                "--presence-context-size" => {
                    ensure!(
                        presence_context_size.is_none(),
                        "--presence-context-size specified twice"
                    );
                    presence_context_size = Some(nonnegative(
                        args.next()
                            .context("--presence-context-size requires a value")?,
                        &arg,
                    )?);
                }
                "--seed" => {
                    ensure!(seed.is_none(), "--seed specified twice");
                    seed = Some(
                        args.next()
                            .context("--seed requires a value")?
                            .parse()
                            .context("--seed requires an unsigned 64-bit integer")?,
                    );
                }
                _ if arg.starts_with('-') => {
                    bail!("unknown option: {arg}\nUse moespresso generate --help for options.")
                }
                _ => {
                    ensure!(package.is_none(), "unexpected positional argument: {arg}");
                    package = Some(arg);
                }
            }
        }
        ensure!(
            expert_capacity.is_none_or(|capacity| (10..=512).contains(&capacity)),
            "expert capacity must be 10..=512"
        );
        Ok(Command::Generate(GenerateArgs {
            package: package.context("missing PACKAGE")?,
            prompt: prompt.context("missing --prompt TEXT")?,
            thinking,
            max_tokens,
            expert_capacity,
            max_memory_bytes,
            max_context,
            cache_routing,
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            presence_context_size,
            seed,
        }))
    }

    fn parse_serve_args(mut args: impl Iterator<Item = String>) -> Result<Command> {
        let mut package = None;
        let mut port = 8080;
        let mut expert_capacity = None;
        let mut max_memory_bytes = None;
        let mut prompt_cache_bytes = None;
        let mut max_context = None;
        let mut thinking = None;
        let mut cache_routing = RoutingMode::CachePrior;
        let mut no_adaptive_memory = false;
        let mut no_disk_cache = false;
        let mut disk_cache_dir = None;
        let mut disk_cache_bytes = None;
        let mut disk_cache_mode = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => return Ok(Command::Help(help::SERVE)),
                "--no-disk-cache" => {
                    ensure!(!no_disk_cache, "--no-disk-cache specified twice");
                    no_disk_cache = true;
                }
                "--disk-cache-dir" => {
                    ensure!(disk_cache_dir.is_none(), "--disk-cache-dir specified twice");
                    let path = args.next().context("--disk-cache-dir requires a path")?;
                    ensure!(
                        !path.is_empty(),
                        "--disk-cache-dir requires a nonempty path"
                    );
                    disk_cache_dir = Some(std::path::PathBuf::from(path));
                }
                "--disk-cache-mode" => {
                    ensure!(
                        disk_cache_mode.is_none(),
                        "--disk-cache-mode specified twice"
                    );
                    disk_cache_mode = Some(match args.next().as_deref() {
                        Some("prefix") => DiskCacheMode::Prefix,
                        Some("full") => DiskCacheMode::Full,
                        _ => bail!("--disk-cache-mode must be prefix or full"),
                    });
                }
                "--disk-cache-gb" => {
                    ensure!(
                        disk_cache_bytes.is_none(),
                        "--disk-cache-gb specified twice"
                    );
                    disk_cache_bytes = Some(gib_bytes(
                        args.next().context("--disk-cache-gb requires a value")?,
                        "--disk-cache-gb",
                    )?);
                }
                "--no-adaptive-memory" => {
                    ensure!(!no_adaptive_memory, "--no-adaptive-memory specified twice");
                    no_adaptive_memory = true;
                }
                "--port" => {
                    port = args
                        .next()
                        .context("--port requires a value")?
                        .parse()
                        .context("--port requires a port number 1..=65535")?;
                    ensure!(port != 0, "--port requires a port number 1..=65535");
                }
                "--expert-capacity" => {
                    ensure!(
                        expert_capacity.is_none() && max_memory_bytes.is_none(),
                        "--expert-capacity and --max-memory-gb are mutually exclusive"
                    );
                    expert_capacity = Some(positive(
                        args.next().context("--expert-capacity requires a value")?,
                        &arg,
                    )?);
                }
                "--max-memory-gb" => {
                    ensure!(
                        max_memory_bytes.is_none() && expert_capacity.is_none(),
                        "--max-memory-gb and --expert-capacity are mutually exclusive"
                    );
                    max_memory_bytes = Some(gib_bytes(
                        args.next().context("--max-memory-gb requires a value")?,
                        "--max-memory-gb",
                    )?);
                }
                "--prompt-cache-gb" => {
                    ensure!(
                        prompt_cache_bytes.is_none(),
                        "--prompt-cache-gb may be specified once"
                    );
                    prompt_cache_bytes = Some(prompt_cache_gib_bytes(
                        args.next().context("--prompt-cache-gb requires a value")?,
                    )?);
                }
                "--max-context" => {
                    max_context = Some(positive(
                        args.next().context("--max-context requires a value")?,
                        &arg,
                    )?);
                }
                "--thinking" => {
                    ensure!(thinking.is_none(), "--thinking may be specified once");
                    thinking = Some(
                        match args.next().context("--thinking requires a value")?.as_str() {
                            "on" => true,
                            "off" => false,
                            _ => bail!("--thinking must be on or off"),
                        },
                    );
                }
                "--cache-routing" => {
                    cache_routing = match args.next().as_deref() {
                        Some("off") => RoutingMode::Unbiased,
                        Some("prefer-resident") => RoutingMode::CachePrior,
                        _ => bail!("--cache-routing must be off or prefer-resident"),
                    };
                }
                _ if arg.starts_with('-') => {
                    bail!("unknown option: {arg}\nUse moespresso serve --help for options.")
                }
                _ => {
                    ensure!(package.is_none(), "unexpected positional argument: {arg}");
                    package = Some(arg);
                }
            }
        }
        ensure!(
            expert_capacity.is_none_or(|capacity| (10..=512).contains(&capacity)),
            "expert capacity must be 10..=512"
        );
        serve_cache_bytes(expert_capacity, prompt_cache_bytes)?;
        ensure!(
            (disk_cache_dir.is_none() && disk_cache_bytes.is_none() && disk_cache_mode.is_none())
                || (!no_disk_cache && expert_capacity.is_none()),
            "disk-cache options conflict with --no-disk-cache and --expert-capacity"
        );
        Ok(Command::Serve(ServeArgs {
            package: package.context("missing PACKAGE")?,
            port,
            expert_capacity,
            max_memory_bytes,
            prompt_cache_bytes,
            max_context,
            thinking: thinking.unwrap_or(true),
            cache_routing,
            adaptive_memory: !no_adaptive_memory && expert_capacity.is_none(),
            no_disk_cache,
            disk_cache_dir,
            disk_cache_bytes,
            disk_cache_mode,
        }))
    }

    pub fn run() -> Result<()> {
        match parse_args(std::env::args().skip(1))? {
            Command::Help(text) => {
                println!("{text}");
                Ok(())
            }
            Command::Version => {
                println!(concat!("moespresso ", env!("CARGO_PKG_VERSION")));
                Ok(())
            }
            Command::CheckAne => crate::check_ane::run(),
            Command::Generate(args) => run_generate(args),
            Command::Serve(args) => run_serve(args),
            Command::Speed(args) => speed::run(args),
        }
    }

    fn run_serve(args: ServeArgs) -> Result<()> {
        let package = Package::open(&args.package)?;
        let resolved_disk = crate::disk_config::resolve(
            &package,
            args.expert_capacity.is_some(),
            args.no_disk_cache,
            args.disk_cache_dir.as_deref(),
            args.disk_cache_bytes,
            args.disk_cache_mode,
        )?;
        if let Some(reason) = resolved_disk.unavailable {
            eprintln!(
                "default disk cache location unavailable; serving without disk caching: {reason}"
            );
        }
        let mut disk = resolved_disk.disk;
        let config = QwenConfig::from_package(&package)?;
        let max_context = serve_context(args.max_context, config.max_positions)?;
        let retained = serve_cache_bytes(args.expert_capacity, args.prompt_cache_bytes)?;
        let disk_overlap = disk.as_ref().map_or(0, |disk| disk.max_io_buffer_bytes);
        let host_overlap = retained
            .checked_add(disk_overlap)
            .context("cache host overlap overflow")?;
        let resolved_capacity = resolve_capacity(
            &package,
            max_context,
            args.expert_capacity,
            args.max_memory_bytes,
            retained,
            disk_overlap,
            args.adaptive_memory,
        )?;
        let expert_capacity = resolved_capacity.expert_capacity;
        let prefill_capacity = resolved_capacity.prefill_capacity;
        let control_budget = resolved_capacity.control_budget;
        if let Some(options) = &disk
            && let Err(error) = crate::disk_config::prepare_root(&options.root)
        {
            if options.required {
                return Err(error).context("explicit disk cache root unavailable");
            }
            eprintln!(
                "default disk cache root unavailable; serving without disk caching: {error:#}"
            );
            disk = None;
        }
        serve_with_prefill_capacity(
            &args.package,
            ServeOptions {
                port: args.port,
                expert_capacity,
                max_context,
                thinking: args.thinking,
                decode_mode: args.cache_routing,
                expert_capacity_control: control_budget,
                adaptive_memory: resolved_capacity.adaptive_memory,
                ane_reserved_bytes: resolved_capacity.ane_reserved_bytes,
                cache: (retained > 0 || disk.is_some()).then_some(ServeCacheOptions {
                    memory_retained_bytes: retained,
                    host_overlap_bytes: host_overlap,
                    disk,
                }),
            },
            prefill_capacity,
        )
    }

    fn run_generate(args: GenerateArgs) -> Result<()> {
        let package = Package::open(&args.package)?;
        let resolved_capacity = resolve_capacity(
            &package,
            args.max_context,
            args.expert_capacity,
            args.max_memory_bytes,
            0,
            0,
            false,
        )?;
        let expert_capacity = resolved_capacity.expert_capacity;
        let prefill_capacity = resolved_capacity.prefill_capacity;
        let mut sampling = SamplingOptions::from_package(&package)?;
        if let Some(value) = args.temperature {
            sampling.temperature = value;
        }
        if let Some(value) = args.top_p {
            sampling.top_p = value;
        }
        if let Some(value) = args.top_k {
            sampling.top_k = value;
        }
        if let Some(value) = args.min_p {
            sampling.min_p = value;
        }
        if let Some(value) = args.presence_penalty {
            sampling.presence_penalty = value;
        }
        if let Some(value) = args.presence_context_size {
            sampling.presence_context_size = value;
        }
        let seed = match args.seed {
            Some(seed) => seed,
            None if sampling.temperature == 0.0 => 0,
            None => OsRng
                .try_next_u64()
                .map_err(|err| anyhow!("operating-system seed failed: {err}"))?,
        };
        let _ = Sampler::new(sampling, seed)?;
        let tokenizer = TextTokenizer::load(&package)?;
        let prompt = tokenizer.render(
            &[json!({"role": "user", "content": args.prompt})],
            None,
            &ChatOptions {
                thinking: args.thinking,
                preserve_thinking: true,
                effort: ReasoningEffort::Medium,
            },
        )?;
        ensure!(
            prompt.tokens.len() <= args.max_context,
            "rendered prompt has {} tokens, exceeding --max-context {}",
            prompt.tokens.len(),
            args.max_context
        );
        eprintln!(
            "☕ loading {}: prompt={} tokens, expert_capacity={}, max_context={}, decode_routing={:?}, sampling={}, temperature={}, top_p={}, top_k={}, min_p={}, presence_penalty={}, presence_context_size={}, seed={}",
            package.artifact_id,
            prompt.tokens.len(),
            expert_capacity,
            args.max_context,
            args.cache_routing,
            if sampling.temperature == 0.0 {
                "greedy"
            } else {
                "native-chaCha8"
            },
            sampling.temperature,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
            sampling.presence_penalty,
            sampling.presence_context_size,
            seed,
        );
        let mut engine = Engine::load_with_prefill_capacity(
            &package,
            expert_capacity,
            args.max_context,
            prefill_capacity,
        )?;
        if resolved_capacity.ane_reserved_bytes > 0 {
            engine.prefer_ane(&package)?;
        }
        let stdout = io::stdout();
        let mut output = stdout.lock();
        let generated = generate(
            &mut engine,
            &tokenizer,
            &prompt.tokens,
            GenerationOptions {
                max_tokens: args.max_tokens,
                max_context: args.max_context,
                decode_mode: args.cache_routing,
                sampling,
                seed,
            },
            |text| {
                output.write_all(text.as_bytes())?;
                output.flush()?;
                Ok(())
            },
            || false,
        );
        let cleanup = engine.close_ane();
        let stats = match (generated, cleanup) {
            (Ok(stats), Ok(())) => stats,
            (Err(error), Ok(())) | (Ok(_), Err(error)) => return Err(error),
            (Err(error), Err(cleanup)) => {
                return Err(error.context(format!("ANE cleanup also failed: {cleanup:#}")));
            }
        };
        eprintln!(
            "finish={:?} prompt_consumed={} generated={} frontier={} changed_routes={}",
            stats.finish,
            stats.prompt_consumed,
            stats.generated_tokens,
            stats.frontier,
            stats.changed_routes,
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn ane_candidates_match_initial_workspace_and_full_residency() {
            assert!(validate_ane_candidate(Some(8192), 512, 8192, 131072).is_ok());
            assert!(validate_ane_candidate(Some(8192), 128, 8192, 131072).is_err());
            assert!(validate_ane_candidate(Some(8192), 512, 4096, 131072).is_err());
            assert!(validate_ane_candidate(Some(8192), 512, 8192, 4096).is_err());
            for capacity in [64, 128, 512] {
                for workspace in [2048, 4096] {
                    assert!(
                        validate_ane_candidate(Some(4096), capacity, workspace, 131072).is_ok()
                    );
                }
            }
            assert!(validate_ane_candidate(Some(4096), 128, 8192, 131072).is_err());
            assert!(validate_ane_candidate(Some(4096), 128, 512, 131072).is_err());
            assert!(validate_ane_candidate(Some(4096), 128, 2048, 2048).is_err());
            assert!(validate_ane_candidate(Some(2048), 128, 2048, 131072).is_err());
            assert!(validate_ane_candidate(None, 128, 512, 2048).is_ok());
        }

        #[test]
        fn speed_command_uses_bounded_client_options() {
            let Command::Speed(defaults) = parse_args(["speed"].map(str::to_owned)).unwrap() else {
                panic!("expected speed command");
            };
            assert_eq!(
                defaults,
                speed::SpeedOptions::parse(std::iter::empty()).unwrap()
            );
            assert!(matches!(
                parse_args(
                    [
                        "speed",
                        "--url",
                        "https://example.org/",
                        "--max-tokens",
                        "8",
                        "--stream"
                    ]
                    .map(str::to_owned)
                )
                .unwrap(),
                Command::Speed(_)
            ));
            assert!(matches!(
                parse_args(["speed", "--help"].map(str::to_owned)).unwrap(),
                Command::Help(help::SPEED)
            ));
        }

        #[test]
        fn parses_bounded_defaults_and_explicit_decode_policy() {
            let Command::Generate(defaults) =
                parse_args(["generate", "model", "--prompt", "Hi"].map(str::to_owned)).unwrap()
            else {
                panic!("expected generate command");
            };
            assert_eq!(defaults.expert_capacity, None);
            assert_eq!(defaults.max_memory_bytes, None);
            assert_eq!(defaults.max_context, 4096);
            assert_eq!(defaults.max_tokens, 128);
            assert!(defaults.thinking);
            assert_eq!(defaults.cache_routing, RoutingMode::CachePrior);

            let Command::Generate(overridden) = parse_args(
                [
                    "generate",
                    "--prompt",
                    "hello",
                    "--no-thinking",
                    "--max-tokens",
                    "2",
                    "--expert-capacity",
                    "10",
                    "--max-context",
                    "256",
                    "--cache-routing",
                    "off",
                    "--temperature",
                    "0.8",
                    "--top-p",
                    "0.95",
                    "--top-k",
                    "24",
                    "--min-p",
                    "0.02",
                    "--presence-penalty",
                    "0.6",
                    "--presence-context-size",
                    "0",
                    "--seed",
                    "123",
                    "model",
                ]
                .map(str::to_owned),
            )
            .unwrap() else {
                panic!("expected generate command");
            };
            assert_eq!(overridden.max_tokens, 2);
            assert_eq!(overridden.expert_capacity, Some(10));
            assert_eq!(overridden.max_memory_bytes, None);
            assert_eq!(overridden.max_context, 256);
            assert!(!overridden.thinking);
            assert_eq!(overridden.cache_routing, RoutingMode::Unbiased);
            assert_eq!(overridden.temperature, Some(0.8));
            assert_eq!(overridden.top_p, Some(0.95));
            assert_eq!(overridden.top_k, Some(24));
            assert_eq!(overridden.min_p, Some(0.02));
            assert_eq!(overridden.presence_penalty, Some(0.6));
            assert_eq!(overridden.presence_context_size, Some(0));
            assert_eq!(overridden.seed, Some(123));

            let Command::Generate(single) = parse_args(
                ["generate", "model", "--prompt", "x", "--max-context", "1"].map(str::to_owned),
            )
            .unwrap() else {
                panic!("expected generate command");
            };
            assert_eq!(single.max_context, 1);

            let Command::Generate(memory) = parse_args(
                [
                    "generate",
                    "model",
                    "--prompt",
                    "Hi",
                    "--max-memory-gb",
                    "1.5",
                ]
                .map(str::to_owned),
            )
            .unwrap() else {
                panic!("expected generate command");
            };
            assert_eq!(memory.max_memory_bytes, Some(3 * GIB_BYTES / 2));
        }

        #[test]
        fn rejects_missing_or_invalid_generation_arguments() {
            for args in [
                vec!["generate", "model"],
                vec!["generate", "model", "--prompt", "x", "--max-tokens", "0"],
                vec![
                    "generate",
                    "model",
                    "--prompt",
                    "x",
                    "--cache-routing",
                    "auto",
                ],
                vec![
                    "generate",
                    "model",
                    "--prompt",
                    "x",
                    "--expert-capacity",
                    "9",
                ],
                vec!["generate", "model", "--prompt", "x", "extra"],
                vec!["generate", "model", "--prompt", "x", "--temperature", "NaN"],
                vec!["generate", "model", "--prompt", "x", "--top-k", "-1"],
                vec!["generate", "model", "--prompt", "x", "--seed", "-1"],
                vec![
                    "generate", "model", "--prompt", "x", "--seed", "2", "--seed", "3",
                ],
            ] {
                assert!(parse_args(args.into_iter().map(str::to_owned)).is_err());
            }
            for flags in [
                "--max-memory-gb 0",
                "--max-memory-gb NaN",
                "--max-memory-gb 1e30",
                "--max-memory-gb 1e-20",
                "--max-memory-gb 2 --expert-capacity 12",
                "--expert-capacity 12 --max-memory-gb 2",
                "--max-memory-gb 2 --max-memory-gb 3",
            ] {
                let args = ["generate", "model", "--prompt", "x"]
                    .into_iter()
                    .chain(flags.split_whitespace())
                    .map(str::to_owned);
                assert!(parse_args(args).is_err(), "accepted invalid flags: {flags}");
            }
        }

        #[test]
        fn disk_cache_location_and_budget_validate_before_loading() {
            let Command::Serve(args) = parse_args(
                [
                    "serve",
                    "model",
                    "--disk-cache-dir",
                    "cache",
                    "--disk-cache-gb",
                    "2.5",
                ]
                .map(str::to_owned),
            )
            .unwrap() else {
                panic!("expected serve command")
            };
            assert_eq!(args.disk_cache_dir, Some(std::path::PathBuf::from("cache")));
            assert_eq!(args.disk_cache_bytes, Some(5 * GIB_BYTES / 2));
            for flags in [
                "--disk-cache-gb 0",
                "--disk-cache-gb NaN",
                "--disk-cache-gb 1e30",
                "--disk-cache-gb -1",
                "--disk-cache-gb 2 --disk-cache-gb 3",
                "--disk-cache-dir a --disk-cache-dir b",
                "--disk-cache-dir",
                "--disk-cache-dir cache --no-disk-cache",
                "--disk-cache-gb 2 --no-disk-cache",
                "--expert-capacity 128 --disk-cache-dir cache",
                "--disk-cache-gb 2 --expert-capacity 128",
                "--disk-cache-mode invalid",
                "--disk-cache-mode",
                "--disk-cache-mode full --disk-cache-mode prefix",
                "--disk-cache-mode prefix --no-disk-cache",
                "--disk-cache-mode full --expert-capacity 128",
            ] {
                assert!(
                    parse_args(
                        ["serve", "model"]
                            .into_iter()
                            .chain(flags.split_whitespace())
                            .map(str::to_owned)
                    )
                    .is_err(),
                    "accepted {flags}"
                );
            }
        }

        #[test]
        fn disk_cache_cli_is_independent_of_memory_cache() {
            for memory in [None, Some("0"), Some("0.5")] {
                let mut input = vec!["serve", "model", "--no-disk-cache"];
                if let Some(value) = memory {
                    input.extend(["--prompt-cache-gb", value]);
                }
                let Command::Serve(args) =
                    parse_args(input.into_iter().map(str::to_owned)).unwrap()
                else {
                    panic!("expected serve command");
                };
                assert!(args.no_disk_cache);
                assert_eq!(
                    serve_cache_bytes(args.expert_capacity, args.prompt_cache_bytes).unwrap(),
                    match memory {
                        None => GIB_BYTES,
                        Some("0") => 0,
                        _ => GIB_BYTES / 2,
                    }
                );
            }
            assert!(
                parse_args(
                    ["serve", "model", "--no-disk-cache", "--no-disk-cache"].map(str::to_owned)
                )
                .is_err()
            );
            let Command::Serve(defaults) =
                parse_args(["serve", "model"].map(str::to_owned)).unwrap()
            else {
                unreachable!()
            };
            assert!(!defaults.no_disk_cache);
            assert_eq!(
                defaults.disk_cache_mode.unwrap_or_default(),
                DiskCacheMode::Prefix
            );
            for (name, mode) in [
                ("prefix", DiskCacheMode::Prefix),
                ("full", DiskCacheMode::Full),
            ] {
                let Command::Serve(selected) =
                    parse_args(["serve", "model", "--disk-cache-mode", name].map(str::to_owned))
                        .unwrap()
                else {
                    unreachable!()
                };
                assert_eq!(selected.disk_cache_mode, Some(mode));
                assert_eq!(
                    serve_cache_bytes(None, selected.prompt_cache_bytes).unwrap(),
                    GIB_BYTES
                );
            }
        }

        #[test]
        fn help_is_command_specific_without_a_package() {
            for (command, expected) in [
                ("serve", help::SERVE),
                ("generate", help::GENERATE),
                ("speed", help::SPEED),
                ("check-ane", help::CHECK_ANE),
            ] {
                for flag in ["--help", "-h"] {
                    assert_eq!(
                        parse_args([command, flag].map(str::to_owned)).unwrap(),
                        Command::Help(expected)
                    );
                }
            }
            assert_eq!(
                parse_args(["--help"].map(str::to_owned)).unwrap(),
                Command::Help(help::OVERVIEW)
            );
        }

        #[test]
        fn ane_check_needs_no_package_and_rejects_configuration_arguments() {
            assert_eq!(
                parse_args(["check-ane"].map(str::to_owned)).unwrap(),
                Command::CheckAne
            );
            for argument in ["model", "--timeout", "--expert-capacity", "--unknown"] {
                assert!(parse_args(["check-ane", argument].map(str::to_owned)).is_err());
            }
        }

        #[test]
        fn serve_defaults_resolve_after_package_validation() {
            let Command::Serve(args) = parse_args(["serve", "model"].map(str::to_owned)).unwrap()
            else {
                unreachable!()
            };
            assert_eq!(args.port, 8080);
            assert!(args.thinking);
            assert_eq!(args.max_context, None);
            assert_eq!(
                serve_cache_bytes(args.expert_capacity, args.prompt_cache_bytes).unwrap(),
                GIB_BYTES
            );
            for (package_max, expected) in
                [(1, 1), (4096, 4096), (131072, 131072), (262144, 131072)]
            {
                assert_eq!(serve_context(None, package_max).unwrap(), expected);
                assert_eq!(
                    serve_context(Some(package_max), package_max).unwrap(),
                    package_max
                );
                assert!(serve_context(Some(package_max + 1), package_max).is_err());
            }
            assert!(serve_context(None, 0).is_err());
            assert!(serve_context(None, 262145).is_err());
            assert!(serve_context(Some(0), 262144).is_err());
            assert_eq!(serve_context(Some(16384), 262144).unwrap(), 16384);
        }

        #[test]
        fn serve_cache_overrides_preserve_fixed_capacity_bypass() {
            for sizing in ["", "--max-memory-gb 20", "--expert-capacity 12"] {
                for cache in ["", "--prompt-cache-gb 0", "--prompt-cache-gb 0.5"] {
                    let parsed = parse_args(
                        ["serve", "model"]
                            .into_iter()
                            .chain(sizing.split_whitespace())
                            .chain(cache.split_whitespace())
                            .map(str::to_owned),
                    );
                    let fixed = sizing.contains("expert-capacity");
                    if fixed && cache.ends_with("0.5") {
                        assert!(parsed.is_err());
                        continue;
                    }
                    let Command::Serve(args) = parsed.unwrap() else {
                        unreachable!()
                    };
                    let bytes =
                        serve_cache_bytes(args.expert_capacity, args.prompt_cache_bytes).unwrap();
                    let expected = if fixed || cache.ends_with(" 0") {
                        0
                    } else if cache.ends_with("0.5") {
                        GIB_BYTES / 2
                    } else {
                        GIB_BYTES
                    };
                    assert_eq!(bytes, expected);
                    assert_eq!(args.expert_capacity, fixed.then_some(12));
                    assert_eq!(args.max_context, None);
                }
            }
            assert!(
                parse_args(
                    [
                        "serve",
                        "model",
                        "--prompt-cache-gb",
                        "1",
                        "--expert-capacity",
                        "12"
                    ]
                    .map(str::to_owned)
                )
                .is_err()
            );
        }

        #[test]
        fn cache_allowance_consumes_budget_without_reducing_context() {
            use moespresso::memory::{CapacityInputs, plan_capacity};
            let context = serve_context(None, 262144).unwrap();
            let cache = serve_cache_bytes(None, None).unwrap();
            let inputs = CapacityInputs {
                process_budget_bytes: 20 * GIB_BYTES,
                user_memory_limit_bytes: None,
                fixed_nonexpert_bytes: 4 * GIB_BYTES,
                context_reserve_bytes: context as u64 * 8192,
                bounded_workspace_bytes: Some(GIB_BYTES),
                packed_expert_bytes_per_slot: GIB_BYTES,
                minimum_slots: 10,
                max_slots: 512,
            };
            let without = plan_capacity(inputs).unwrap();
            let with = plan_capacity(CapacityInputs {
                fixed_nonexpert_bytes: inputs.fixed_nonexpert_bytes + cache,
                ..inputs
            })
            .unwrap();
            assert_eq!(cache, GIB_BYTES);
            assert_eq!(without.slots, with.slots + 1);
            assert_eq!(without.accounted_peak_bytes, with.accounted_peak_bytes);
            assert!(
                plan_capacity(CapacityInputs {
                    process_budget_bytes: 16 * GIB_BYTES,
                    fixed_nonexpert_bytes: inputs.fixed_nonexpert_bytes + cache,
                    ..inputs
                })
                .is_err()
            );
            assert_eq!(context, 131072);
            assert_eq!(serve_context(None, 262144).unwrap(), context);
        }

        #[test]
        fn serving_adapts_automatically_and_has_a_static_override() {
            for (flags, expected, routing) in [
                ("", true, RoutingMode::CachePrior),
                ("--max-memory-gb 20", true, RoutingMode::CachePrior),
                ("--no-adaptive-memory", false, RoutingMode::CachePrior),
                (
                    "--no-adaptive-memory --cache-routing off",
                    false,
                    RoutingMode::Unbiased,
                ),
                ("--expert-capacity 10", false, RoutingMode::CachePrior),
            ] {
                let args = format!("serve model {flags}");
                let Command::Serve(parsed) =
                    parse_args(args.split_whitespace().map(str::to_owned)).unwrap()
                else {
                    panic!("expected serve");
                };
                assert_eq!(parsed.adaptive_memory, expected);
                assert_eq!(parsed.cache_routing, routing);
            }
            assert!(
                parse_args(
                    [
                        "serve",
                        "model",
                        "--no-adaptive-memory",
                        "--no-adaptive-memory"
                    ]
                    .map(str::to_owned)
                )
                .is_err()
            );
        }

        #[test]
        fn adaptive_mode_respects_full_residency_and_the_64_slot_floor() {
            assert_eq!(adaptive_mode(512, true), AdaptiveMode::FullResident);
            for slots in 64..512 {
                assert_eq!(adaptive_mode(slots, true), AdaptiveMode::Enabled);
                assert_eq!(adaptive_mode(slots, false), AdaptiveMode::Disabled);
            }
            for slots in 10..64 {
                assert_eq!(adaptive_mode(slots, true), AdaptiveMode::BelowFloor);
                assert!(!adaptive_mode(slots, true).enabled());
                assert_eq!(
                    adaptive_mode(slots, true).reason(),
                    "below-64-slot-adaptive-floor"
                );
            }
        }

        #[test]
        fn parses_loopback_serve_command() {
            let Command::Serve(args) = parse_args(
                [
                    "serve",
                    "model",
                    "--port",
                    "8765",
                    "--expert-capacity",
                    "10",
                    "--max-context",
                    "1024",
                    "--cache-routing",
                    "off",
                ]
                .map(str::to_owned),
            )
            .unwrap() else {
                panic!("expected serve command")
            };
            assert_eq!(args.package, "model");
            assert_eq!(args.port, 8765);
            assert_eq!(args.expert_capacity, Some(10));
            assert_eq!(args.max_memory_bytes, None);
            assert_eq!(args.prompt_cache_bytes, None);
            assert_eq!(args.max_context, Some(1024));
            assert!(args.thinking);
            assert_eq!(args.cache_routing, RoutingMode::Unbiased);
            assert!(parse_args(["serve", "model", "--port", "0"].map(str::to_owned)).is_err());

            let Command::Serve(memory) =
                parse_args(["serve", "model", "--max-memory-gb", "2"].map(str::to_owned)).unwrap()
            else {
                panic!("expected serve command");
            };
            assert_eq!(memory.max_memory_bytes, Some(2 * GIB_BYTES));
            assert_eq!(memory.prompt_cache_bytes, None);
            let Command::Serve(cached) = parse_args(
                [
                    "serve",
                    "model",
                    "--max-memory-gb",
                    "20",
                    "--prompt-cache-gb",
                    "0.5",
                ]
                .map(str::to_owned),
            )
            .unwrap() else {
                panic!("expected serve command");
            };
            assert_eq!(cached.max_memory_bytes, Some(20 * GIB_BYTES));
            assert_eq!(cached.prompt_cache_bytes, Some(GIB_BYTES / 2));
            assert!(
                parse_args(
                    [
                        "serve",
                        "model",
                        "--expert-capacity",
                        "12",
                        "--max-memory-gb",
                        "2"
                    ]
                    .map(str::to_owned)
                )
                .is_err()
            );
            for flags in [
                "--max-memory-gb 20 --prompt-cache-gb",
                "--max-memory-gb 20 --prompt-cache-gb -1",
                "--max-memory-gb 20 --prompt-cache-gb NaN",
                "--max-memory-gb 20 --prompt-cache-gb Inf",
                "--max-memory-gb 20 --prompt-cache-gb 1e30",
                "--max-memory-gb 20 --prompt-cache-gb 1e-20",
                "--max-memory-gb 20 --prompt-cache-gb 1 --prompt-cache-gb 2",
                "--expert-capacity 12 --prompt-cache-gb 1",
            ] {
                let args = ["serve", "model"]
                    .into_iter()
                    .chain(flags.split_whitespace())
                    .map(str::to_owned);
                assert!(parse_args(args).is_err(), "accepted invalid flags: {flags}");
            }
            assert!(
                parse_args(
                    [
                        "generate",
                        "model",
                        "--prompt",
                        "x",
                        "--prompt-cache-gb",
                        "1"
                    ]
                    .map(str::to_owned)
                )
                .is_err()
            );
            assert!(
                parse_args(
                    [
                        "serve",
                        "model",
                        "--max-memory-gb",
                        "2",
                        "--expert-capacity",
                        "12"
                    ]
                    .map(str::to_owned)
                )
                .is_err()
            );
        }

        #[test]
        fn serve_thinking_flag_defaults_on_and_accepts_one_on_or_off_value() {
            for (value, expected) in [("on", true), ("off", false)] {
                let Command::Serve(args) =
                    parse_args(["serve", "model", "--thinking", value].map(str::to_owned)).unwrap()
                else {
                    panic!("expected serve command");
                };
                assert_eq!(args.thinking, expected);
            }
            for args in [
                vec!["serve", "model", "--thinking"],
                vec!["serve", "model", "--thinking", "true"],
                vec!["serve", "model", "--thinking", "off", "--thinking", "on"],
            ] {
                assert!(parse_args(args.into_iter().map(str::to_owned)).is_err());
            }
        }

        #[test]
        fn user_memory_cap_only_clamps_the_automatic_safety_policy() {
            let observed = HostMemoryObservation {
                physical_ram_bytes: 64 * GIB_BYTES,
                free_inactive_bytes: 20 * GIB_BYTES,
                gpu_wired_limit_bytes: Some(28 * GIB_BYTES),
            };
            let automatic = automatic_host_budget(observed, 30 * GIB_BYTES).unwrap();
            assert_eq!(automatic.process_budget_bytes, 17 * GIB_BYTES);
            for limit in [1, 10 * GIB_BYTES, 17 * GIB_BYTES, 25 * GIB_BYTES, u64::MAX] {
                let capped = host_budget(limit, observed, 30 * GIB_BYTES).unwrap();
                assert_eq!(
                    capped.process_budget_bytes,
                    limit.min(automatic.process_budget_bytes)
                );
            }
            assert!(host_budget(0, observed, 30 * GIB_BYTES).is_err());
            assert!(host_budget(1, observed, AUTO_METAL_RESERVE_BYTES).is_err());
        }

        #[test]
        fn absent_capacity_is_auto_but_explicit_twelve_is_not() {
            for command in ["generate", "serve"] {
                let mut base = vec![command, "model"];
                if command == "generate" {
                    base.extend(["--prompt", "x"]);
                }
                for explicit in [false, true] {
                    let mut args = base.clone();
                    if explicit {
                        args.extend(["--expert-capacity", "12"]);
                    }
                    let (slots, budget, context, routing) =
                        match parse_args(args.into_iter().map(str::to_owned)).unwrap() {
                            Command::Generate(args) => (
                                args.expert_capacity,
                                args.max_memory_bytes,
                                Some(args.max_context),
                                args.cache_routing,
                            ),
                            Command::Serve(args) => {
                                assert_eq!(args.prompt_cache_bytes, None);
                                (
                                    args.expert_capacity,
                                    args.max_memory_bytes,
                                    args.max_context,
                                    args.cache_routing,
                                )
                            }
                            _ => unreachable!(),
                        };
                    assert_eq!(slots, explicit.then_some(12));
                    assert_eq!(budget, None);
                    assert_eq!(context, (command == "generate").then_some(4096));
                    assert_eq!(routing, RoutingMode::CachePrior);
                }
                for flags in [
                    "--expert-capacity 12 --expert-capacity 12",
                    "--expert-capacity 12 --max-memory-gb 20",
                    "--max-memory-gb 20 --expert-capacity 12",
                ] {
                    let args = base
                        .iter()
                        .copied()
                        .chain(flags.split_whitespace())
                        .map(str::to_owned);
                    assert!(parse_args(args).is_err());
                }
            }
        }

        #[test]
        fn automatic_ceiling_table_covers_32_and_128_gib_hosts() {
            for (ram, live, metal, wired, expected, source) in [
                (32, 30, 30, None, 27, "physical-minus-5-GiB"),
                (32, 20, 30, None, 17, "free-plus-inactive-minus-3-GiB"),
                (32, 30, 24, None, 23, "Metal-recommended-minus-1-GiB"),
                (32, 30, 30, Some(22), 21, "GPU-wired-limit-minus-1-GiB"),
                (128, 112, 96, None, 95, "Metal-recommended-minus-1-GiB"),
                (
                    128,
                    96,
                    112,
                    Some(120),
                    93,
                    "free-plus-inactive-minus-3-GiB",
                ),
                (128, 128, 128, None, 123, "physical-minus-5-GiB"),
            ] {
                let budget = automatic_host_budget(
                    HostMemoryObservation {
                        physical_ram_bytes: ram * GIB_BYTES,
                        free_inactive_bytes: live * GIB_BYTES,
                        gpu_wired_limit_bytes: wired.map(|v| v * GIB_BYTES),
                    },
                    metal * GIB_BYTES,
                )
                .unwrap();
                assert_eq!(budget.process_budget_bytes, expected * GIB_BYTES);
                assert_eq!(budget.limiting_source, source);
                println!(
                    "auto_policy ram_gib={ram} live_gib={live} metal_gib={metal} wired_gib={wired:?} budget_gib={expected} source={source}"
                );
            }
            let observed = HostMemoryObservation {
                physical_ram_bytes: 32 * GIB_BYTES,
                free_inactive_bytes: 20 * GIB_BYTES,
                gpu_wired_limit_bytes: None,
            };
            assert_eq!(
                host_budget(25 * GIB_BYTES, observed, 24 * GIB_BYTES)
                    .unwrap()
                    .process_budget_bytes,
                17 * GIB_BYTES
            );
            assert_eq!(
                automatic_host_budget(observed, 24 * GIB_BYTES)
                    .unwrap()
                    .process_budget_bytes,
                17 * GIB_BYTES
            );
        }

        #[test]
        fn operating_ceiling_uses_hardware_limits_independently_of_startup_live_pages() {
            for (ram, metal, wired, expected, source) in [
                (32, 30, None, 27, "physical-minus-5-GiB"),
                (32, 24, None, 23, "Metal-recommended-minus-1-GiB"),
                (32, 30, Some(22), 21, "GPU-wired-limit-minus-1-GiB"),
                (128, 96, None, 95, "Metal-recommended-minus-1-GiB"),
                (128, 128, None, 123, "physical-minus-5-GiB"),
            ] {
                for live in [0, 4, 20, ram] {
                    let observed = HostMemoryObservation {
                        physical_ram_bytes: ram * GIB_BYTES,
                        free_inactive_bytes: live * GIB_BYTES,
                        gpu_wired_limit_bytes: wired.map(|value| value * GIB_BYTES),
                    };
                    let operating =
                        operating_host_budget(observed, metal * GIB_BYTES, None).unwrap();
                    assert_eq!(operating.process_budget_bytes, expected * GIB_BYTES);
                    assert_eq!(operating.limiting_source, source);
                    if live == 20 {
                        assert!(
                            automatic_host_budget(observed, metal * GIB_BYTES)
                                .unwrap()
                                .process_budget_bytes
                                < operating.process_budget_bytes
                        );
                    }
                }
            }
        }

        #[test]
        fn explicit_operating_ceiling_keeps_existing_margins_and_applies_the_user_cap() {
            let observed = HostMemoryObservation {
                physical_ram_bytes: 32 * GIB_BYTES,
                free_inactive_bytes: 10 * GIB_BYTES,
                gpu_wired_limit_bytes: Some(22 * GIB_BYTES),
            };
            for (user, metal, wired, expected, source) in [
                (25, 24, Some(22), 21, "GPU-wired-limit-minus-1-GiB"),
                (25, 24, None, 23, "Metal-recommended-minus-1-GiB"),
                (20, 24, Some(22), 20, "explicit-user-limit"),
                (30, 32, None, 27, "physical-minus-5-GiB"),
            ] {
                let operating = operating_host_budget(
                    HostMemoryObservation {
                        gpu_wired_limit_bytes: wired.map(|value| value * GIB_BYTES),
                        ..observed
                    },
                    metal * GIB_BYTES,
                    Some(user * GIB_BYTES),
                )
                .unwrap();
                assert_eq!(operating.process_budget_bytes, expected * GIB_BYTES);
                assert_eq!(operating.limiting_source, source);
            }
            assert_eq!(
                host_budget(20 * GIB_BYTES, observed, 24 * GIB_BYTES)
                    .unwrap()
                    .process_budget_bytes,
                7 * GIB_BYTES
            );
            assert_eq!(
                operating_host_budget(observed, 24 * GIB_BYTES, Some(20 * GIB_BYTES))
                    .unwrap()
                    .process_budget_bytes,
                20 * GIB_BYTES
            );
            assert!(operating_host_budget(observed, 24 * GIB_BYTES, Some(0)).is_err());
            assert!(operating_host_budget(observed, 0, Some(20 * GIB_BYTES)).is_err());
            assert!(operating_host_budget(observed, AUTO_METAL_RESERVE_BYTES, None).is_err());
            assert!(operating_host_budget(observed, 1, Some(20 * GIB_BYTES)).is_err());
            assert!(
                operating_host_budget(
                    HostMemoryObservation {
                        gpu_wired_limit_bytes: Some(0),
                        ..observed
                    },
                    24 * GIB_BYTES,
                    Some(20 * GIB_BYTES)
                )
                .is_err()
            );
            assert!(
                operating_host_budget(
                    HostMemoryObservation {
                        gpu_wired_limit_bytes: Some(AUTO_METAL_RESERVE_BYTES),
                        ..observed
                    },
                    24 * GIB_BYTES,
                    None
                )
                .is_err()
            );
        }

        #[test]
        fn automatic_reserves_reject_unavailable_or_insufficient_observations() {
            let observed = HostMemoryObservation {
                physical_ram_bytes: 32 * GIB_BYTES,
                free_inactive_bytes: 20 * GIB_BYTES,
                gpu_wired_limit_bytes: None,
            };
            for physical in [0, OS_RESERVE_BYTES - 1, OS_RESERVE_BYTES] {
                assert!(
                    automatic_host_budget(
                        HostMemoryObservation {
                            physical_ram_bytes: physical,
                            free_inactive_bytes: physical,
                            ..observed
                        },
                        24 * GIB_BYTES
                    )
                    .is_err()
                );
            }
            for live in [0, AUTO_LIVE_RESERVE_BYTES - 1, AUTO_LIVE_RESERVE_BYTES] {
                assert!(
                    automatic_host_budget(
                        HostMemoryObservation {
                            free_inactive_bytes: live,
                            ..observed
                        },
                        24 * GIB_BYTES
                    )
                    .is_err()
                );
            }
            for ceiling in [0, AUTO_METAL_RESERVE_BYTES - 1, AUTO_METAL_RESERVE_BYTES] {
                assert!(automatic_host_budget(observed, ceiling).is_err());
                assert!(
                    automatic_host_budget(
                        HostMemoryObservation {
                            gpu_wired_limit_bytes: Some(ceiling),
                            ..observed
                        },
                        24 * GIB_BYTES
                    )
                    .is_err()
                );
            }
            assert!(
                automatic_host_budget(
                    HostMemoryObservation {
                        free_inactive_bytes: 33 * GIB_BYTES,
                        ..observed
                    },
                    24 * GIB_BYTES
                )
                .is_err()
            );
            // Missing optional wired information is not fabricated as zero.
            assert!(automatic_host_budget(observed, 24 * GIB_BYTES).is_ok());
        }

        #[test]
        fn decreasing_user_caps_never_increase_the_budget_or_expert_capacity() {
            let observed = HostMemoryObservation {
                physical_ram_bytes: 128 * GIB_BYTES,
                free_inactive_bytes: 110 * GIB_BYTES,
                gpu_wired_limit_bytes: None,
            };
            let mut previous = 512;
            for limit in (7..=110).rev() {
                let budget = host_budget(limit * GIB_BYTES, observed, 96 * GIB_BYTES).unwrap();
                let plan = moespresso::memory::plan_capacity(moespresso::memory::CapacityInputs {
                    process_budget_bytes: budget.process_budget_bytes,
                    user_memory_limit_bytes: None,
                    fixed_nonexpert_bytes: 4 * GIB_BYTES,
                    context_reserve_bytes: GIB_BYTES,
                    bounded_workspace_bytes: Some(GIB_BYTES),
                    packed_expert_bytes_per_slot: GIB_BYTES / 16,
                    minimum_slots: 10,
                    max_slots: 512,
                })
                .unwrap();
                assert!(plan.slots <= previous);
                previous = plan.slots;
            }
        }

        #[test]
        fn automatic_budget_and_affordable_slots_are_monotonic() {
            use moespresso::memory::{CapacityInputs, plan_capacity};
            let mut last_budget = 0;
            let mut last_slots = 0;
            for live in 4..=32 {
                let budget = automatic_host_budget(
                    HostMemoryObservation {
                        physical_ram_bytes: 32 * GIB_BYTES,
                        free_inactive_bytes: live * GIB_BYTES,
                        gpu_wired_limit_bytes: Some(32 * GIB_BYTES),
                    },
                    32 * GIB_BYTES,
                )
                .unwrap();
                assert!(budget.process_budget_bytes >= last_budget);
                last_budget = budget.process_budget_bytes;
                let input = CapacityInputs {
                    process_budget_bytes: budget.process_budget_bytes,
                    user_memory_limit_bytes: None,
                    fixed_nonexpert_bytes: 3 * GIB_BYTES + PROCESS_OVERHEAD_ALLOWANCE_BYTES,
                    context_reserve_bytes: GIB_BYTES / 4,
                    bounded_workspace_bytes: Some(GIB_BYTES / 2),
                    packed_expert_bytes_per_slot: GIB_BYTES / 16,
                    minimum_slots: 10,
                    max_slots: 512,
                };
                match plan_capacity(input) {
                    Ok(plan) => {
                        assert!(plan.slots >= last_slots);
                        assert!((10..=512).contains(&plan.slots));
                        assert!(plan.accounted_peak_bytes <= budget.process_budget_bytes);
                        last_slots = plan.slots;
                    }
                    Err(_) => {
                        assert_eq!(last_slots, 0);
                        assert!(budget.process_budget_bytes < 6 * GIB_BYTES + 3 * GIB_BYTES / 8);
                    }
                }
            }
            assert!(last_slots > 10);
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    cli::run()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("MoEspresso generation requires macOS and Metal");
    std::process::exit(1);
}
