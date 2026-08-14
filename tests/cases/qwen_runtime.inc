/* SPDX-License-Identifier: GPL-2.0-only */

/* Qwen runtime tests register here. */

M3_TEST_CASE(m3_test_qwen_official_plan,
             "official Qwen constants and weight order")
M3_TEST_CASE(m3_test_qwen_cache_layout,
             "Qwen token-major cache layout and bytes")
M3_TEST_CASE(m3_test_qwen_runtime_api_validation,
             "Qwen runtime API validation")
M3_TEST_CASE(m3_test_qwen_rope_oracles,
             "Qwen BF16 half-split RoPE oracles")
M3_TEST_CASE(m3_test_qwen_forward_oracle,
             "Qwen reusable forward numerical oracle")
M3_TEST_CASE(m3_test_qwen_forward_atomicity,
             "Qwen forward cancellation and atomic state")
M3_TEST_CASE(m3_test_qwen_forward_validation,
             "Qwen forward bounds and state transitions")
M3_TEST_CASE(m3_test_qwen_prefill_causal_offset,
             "Qwen prefill causal offset zero")
M3_TEST_CASE(m3_test_qwen_step_causal_offset,
             "Qwen token-step published causal offset")
