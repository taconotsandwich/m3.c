/* SPDX-License-Identifier: GPL-2.0-only */

M3_TEST_CASE(m3_test_flow_binding,
             "flow exact staged binding")
M3_TEST_CASE(m3_test_flow_schedule_contract,
             "flow exact scheduler contract")
M3_TEST_CASE(m3_test_flow_chunk_contract,
             "flow chunk and carry contract")
M3_TEST_CASE(m3_test_flow_fourier_rotary,
             "flow Fourier and partial RoPE")
M3_TEST_CASE(m3_test_flow_layer_noncausal,
             "flow layer noncausal attention")
M3_TEST_CASE(m3_test_flow_layer_feed_forward,
             "flow layer feed-forward order")
M3_TEST_CASE(m3_test_flow_cfg_euler_blend,
             "flow CFG Euler and overlap arithmetic")
M3_TEST_CASE(m3_test_flow_model_rows_and_carry,
             "flow model row order and carry ownership")
M3_TEST_CASE(m3_test_flow_runtime_chunks,
             "flow synthesis chunks and RNG")
M3_TEST_CASE(m3_test_flow_runtime_atomic_retry,
             "flow atomic cancellation and retry")
M3_TEST_CASE(m3_test_flow_runtime_validation,
             "flow runtime validation")
M3_TEST_CASE(m3_test_flow_preflight_live_set,
             "flow aggregate live-set preflight")
