"""Qwen-family runtime graph adapters and served-path optimizations.

Model-specific runtime code for the qwen3_5_moe family lives here, beside the
shared serving, generation, and cache infrastructure in the runtime root. The
first module is the sorted K-quant routed-expert path (`sorted_switch_glu`),
which replaces the stock unsorted SwitchGLU gather matmuls with the sorted-ids,
full-resident, barrier-free route.
"""
