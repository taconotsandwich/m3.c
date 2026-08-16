/* SPDX-License-Identifier: GPL-2.0-only */

M3_TEST_CASE(m3_test_graph_cpu_liveness,
             "graph runtime reuses CPU storage by liveness")
M3_TEST_CASE(m3_test_graph_validation,
             "graph runtime rejects malformed data flow atomically")
M3_TEST_CASE(m3_test_graph_metal_provider,
             "graph runtime selects and executes the Metal provider")
M3_TEST_CASE(m3_test_graph_coreml_provider,
             "graph runtime executes a Core ML partition")
