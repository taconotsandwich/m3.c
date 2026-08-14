/* SPDX-License-Identifier: GPL-2.0-only */

M3_TEST_CASE(m3_test_rvq_binding,
             "RVQ staged binding")
M3_TEST_CASE(m3_test_condition_binding,
             "condition staged binding")
M3_TEST_CASE(m3_test_rvq_decode_schedule,
             "RVQ exact depth schedule")
M3_TEST_CASE(m3_test_rvq_atomic_validation,
             "RVQ atomic validation and cancellation")
M3_TEST_CASE(m3_test_rvq_feedback_oracle,
             "RVQ complete-frame Qwen feedback oracle")
M3_TEST_CASE(m3_test_rvq_feedback_atomic_validation,
             "RVQ feedback atomic validation")
M3_TEST_CASE(m3_test_condition_numeric_resize,
             "condition ordered numeric and resize oracle")
M3_TEST_CASE(m3_test_condition_atomic_validation,
             "condition atomic validation and cancellation")
M3_TEST_CASE(m3_test_rvq_condition_preflight_limits,
             "RVQ and condition aggregate preflight")
