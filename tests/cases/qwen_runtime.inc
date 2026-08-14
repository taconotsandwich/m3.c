/* SPDX-License-Identifier: GPL-2.0-only */

/* Qwen runtime tests register here. */

M3_TEST_CASE(m3_test_qwen_official_plan,
             "official Qwen constants and weight order")
M3_TEST_CASE(m3_test_qwen_cache_layout,
             "Qwen token-major cache layout and bytes")
M3_TEST_CASE(m3_test_qwen_runtime_api_validation,
             "Qwen runtime API validation")
M3_TEST_CASE(m3_test_qwen_semantic_embedding,
             "Qwen semantic code embedding endpoints")
M3_TEST_CASE(m3_test_qwen_rope_oracles,
             "Qwen BF16 half-split RoPE oracles")
M3_TEST_CASE(m3_test_qwen_forward_oracle,
             "Qwen reusable forward numerical oracle")
M3_TEST_CASE(m3_test_qwen_forward_atomicity,
             "Qwen forward cancellation and atomic state")
M3_TEST_CASE(m3_test_qwen_forward_validation,
             "Qwen forward bounds and state transitions")
M3_TEST_CASE(m3_test_qwen_strided_feedback,
             "Qwen padded strided feedback validation")
M3_TEST_CASE(m3_test_qwen_prefill_causal_offset,
             "Qwen prefill causal offset zero")
M3_TEST_CASE(m3_test_qwen_advance_causal_offset,
             "Qwen complete-frame published causal offset")

M3_TEST_CASE(m3_test_semantic_plan_contract,
             "semantic generation exact allocation plan")
M3_TEST_CASE(m3_test_semantic_request_contract,
             "semantic generation request and backend contract")
M3_TEST_CASE(m3_test_semantic_schedule_caps,
             "semantic warmup and capped frame schedules")
M3_TEST_CASE(m3_test_semantic_guidance_row_order,
             "semantic conditional and CFG sampler row order")
M3_TEST_CASE(m3_test_semantic_eos_schedule,
             "semantic EOS attempts and frame publication")
M3_TEST_CASE(m3_test_semantic_atomic_faults,
             "semantic phase fault atomicity")
M3_TEST_CASE(m3_test_semantic_cancellation,
             "semantic checkpoint cancellation atomicity")
M3_TEST_CASE(m3_test_semantic_reduced_metal,
             "semantic reduced real-Metal scheduler")
