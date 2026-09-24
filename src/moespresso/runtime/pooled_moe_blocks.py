"""Model math adapters for the shared pooled MoE scheduler."""

from __future__ import annotations

import time

import mlx.core as mx

from moespresso.runtime import pooled_switchglu as pooled
from moespresso.runtime.pooled_moe import run_pooled_moe


def _run(block, x, *, deepseek, input_ids=None):
    switch = block.switch_mlp
    if block.sharding_group is not None:
        from mlx.nn.layers.distributed import sum_gradients

        x = sum_gradients(block.sharding_group)(x)
    decode = pooled._token_layers(x) == 1
    started = time.perf_counter() if decode else None
    if pooled._ROUTE_TRACE is not None and pooled._ROUTE_TRACE_HIDDEN and decode:
        import numpy as np

        pooled._ROUTE_TRACE.append(
            (
                "hidden",
                switch.gate_proj.pool.layer,
                np.asarray(x).reshape(-1).astype(np.float16),
            )
        )
    if deepseek:
        route_started = time.perf_counter()
        indices, scores = block.gate(x, input_ids=input_ids)
        if decode:
            pooled._record_switch_seconds(
                switch, "router_gate_seconds", time.perf_counter() - route_started
            )
    else:
        gates = mx.softmax(block.gate(x), axis=-1, precise=True)
        indices = mx.argpartition(gates, kth=-block.top_k, axis=-1)[..., -block.top_k :]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        if block.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
    indices = indices.astype(mx.uint32)

    def reduce(output, weights, _indices):
        pooled._record_routed_weighted_sum(switch, weights, out_features=int(output.shape[-1]))
        if deepseek:
            return pooled._deepseek_v4_weighted_sum(output, weights).reshape(x.shape)
        return (output * weights[..., None]).sum(axis=-2)

    def shared(value):
        if deepseek:
            return block.shared_experts(value)
        return mx.sigmoid(block.shared_expert_gate(value)) * block.shared_expert(value)

    def resident(value, ids, weights):
        ready = getattr(switch, "_barrier_free_decode_ready", None)
        if (
            pooled._ROUTE_TRACE is not None
            or ready is None
            or not ready()
        ):
            return None
        fused = getattr(switch, "decode_routed_fused_engaged", None)
        if deepseek and callable(fused) and fused():
            return switch.build_barrier_free_decode_fused(value, ids, weights)
        return reduce(switch.build_barrier_free_decode(value, ids), weights, ids)

    def pipelined(value, ids, weights, *, event_gate):
        fused = getattr(switch, "_decode_routed_fused_ready", None)
        if deepseek and callable(fused) and fused():
            return switch.build_pipelined_fused(value, ids, weights, event_gate=event_gate).reshape(
                x.shape
            )
        return reduce(switch.build_pipelined(value, ids, event_gate=event_gate), weights, ids)

    def direct(value, ids, weights, *, load_ticket):
        weighted = getattr(switch, "weighted_output", None)
        if deepseek and callable(weighted):
            return weighted(value, ids, weights, load_ticket=load_ticket).reshape(x.shape)
        return reduce(switch(value, ids, load_ticket=load_ticket), weights, ids)

    def flush(result):
        if deepseek and bool(getattr(switch, "_all_iqk", False)):
            if switch.commit_iqk_output(result, rows=1):
                switch.barrier_free_decode_flush_calls += 1
        elif (
            pooled._DECODE_FLUSH_LAYERS > 0
            and (int(switch.gate_proj.pool.layer) + 1) % pooled._DECODE_FLUSH_LAYERS == 0
        ):
            pooled._kick_eval(result)
            switch.barrier_free_decode_flush_calls += 1

    result = run_pooled_moe(
        switch,
        x,
        indices,
        scores,
        shared=shared,
        reduce=reduce,
        resident=resident,
        pipelined=pipelined,
        direct=direct,
        flush_resident=flush,
        training=block.training,
        last=block.pipeline_is_last,
        block_started=started,
    )
    if block.sharding_group is not None:
        result = mx.distributed.all_sum(result, group=block.sharding_group)
    return result
