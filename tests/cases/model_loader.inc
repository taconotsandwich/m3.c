/* SPDX-License-Identifier: GPL-2.0-only */

M3_TEST_CASE(m3_test_safetensors_header,
             "inspect supported Safetensors headers")
M3_TEST_CASE(m3_test_safetensors_rejections,
             "reject malformed Safetensors headers")
M3_TEST_CASE(m3_test_modular_model_directory,
             "inspect the official modular directory layout")
M3_TEST_CASE(m3_test_modular_manifest,
             "validate the official modular manifest contract")
M3_TEST_CASE(m3_test_modular_model_index,
             "validate sharded indexes and reject legacy resources")
M3_TEST_CASE(m3_test_official_snapshot_manifest,
             "inspect the current official snapshot manifest")
