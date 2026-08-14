/* SPDX-License-Identifier: GPL-2.0-only */

M3_TEST_CASE(m3_test_runtime_workspace_views,
             "runtime workspace builds exact aligned views")
M3_TEST_CASE(m3_test_runtime_workspace_failures,
             "runtime workspace rejects checked limits")
M3_TEST_CASE(m3_test_runtime_workspace_atomic,
             "runtime workspace replacement is atomic")
M3_TEST_CASE(m3_test_runtime_weight_resolver,
             "runtime weight resolver is exact and atomic")
M3_TEST_CASE(m3_test_runtime_command_executor,
             "runtime command executor reuses checked scratch")
