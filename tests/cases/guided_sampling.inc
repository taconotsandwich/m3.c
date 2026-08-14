/* SPDX-License-Identifier: GPL-2.0-only */

M3_TEST_CASE(m3_test_guided_cfg_distribution,
             "exact classifier-free guided distribution")
M3_TEST_CASE(m3_test_guided_semantic_masks,
             "semantic conditional and common top-k masks")
M3_TEST_CASE(m3_test_guided_boundaries_and_dtypes,
             "guided code boundaries and float dtypes")
M3_TEST_CASE(m3_test_guided_nonfinite_common,
             "guided common nonfinite sanitization")
M3_TEST_CASE(m3_test_guided_validation,
             "guided residual validation and atomicity")
M3_TEST_CASE(m3_test_guided_semantic_validation,
             "guided semantic validation and atomicity")
